/* Inference for Mamba model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Mamba model

typedef struct {
    int n_layers;     // number of layers
    int input_size;   // input dimension (784 for MNIST)
    int dim;          // hidden dimension
    int d_inner;
    int dt_rank;
    int d_state;
    int d_conv;
    int num_classes;  // number of classes (10 for MNIST)
} Config;

typedef struct {
    // input projection
    float* input_proj;     // (dim, input_size)
    // weights for layers
    float* in_proj;        // (layer, 2*d_inner, dim)
    float* conv1d_weight;  // (layer, d_inner, 1, d_conv)
    float* conv1d_bias;    // (layer, d_inner)
    float* x_proj;         // (layer, dt_rank+2*d_state, d_inner)
    float* dt_proj_weight; // (layer, d_inner, dt_rank)
    float* dt_proj_bias;   // (layer, d_inner)
    float* A;              // (layer, d_inner, d_state)
    float* D;              // (layer, d_inner)
    float* out_proj;       // (layer, dim, d_inner)
    float* norm;           // (layer, dim)
    // final rmsnorm
    float* final_norm;     // (dim)
    // classifier weights
    float* classifier;     // (num_classes, dim)
} MambaWeights;

typedef struct {
    // memory reused by all layers
    float* input;        // (dim)
    float* hidden_state; // (dim)
    float* xz;          // (2*d_inner)
    float* x_db;        // (dt_rank+2*d_state)
    float* dt;          // (d_inner)
    float* dA;          // (d_inner, d_state)
    float* dB;          // (d_inner, d_state)
    float* temp;        // (d_inner, d_state)
    float* y;           // (d_inner)
    float* logits;      // (num_classes)
    // internal state
    float* conv_state;  // (n_layers, d_inner, d_conv)
    float* ssm_state;   // (n_layers, d_inner, d_state)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    MambaWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Mamba;

void malloc_run_state(RunState* s, Config* p) {
    // memory reused by all layers
    s->input = malloc(p->dim * sizeof(float));
    s->hidden_state = malloc(p->dim * sizeof(float));
    s->xz = malloc(2 * p->d_inner * sizeof(float));
    s->x_db = malloc((p->dt_rank + 2 * p->d_state) * sizeof(float));
    s->dt = malloc(p->d_inner * sizeof(float));
    s->dA = malloc(p->d_inner * p->d_state * sizeof(float));
    s->dB = malloc(p->d_inner * p->d_state * sizeof(float));
    s->temp = malloc(p->d_inner * p->d_state * sizeof(float));
    s->y = malloc(p->d_inner * sizeof(float));
    s->logits = malloc(p->num_classes * sizeof(float));
    // internal state, separate memory for each layer
    s->conv_state = calloc(p->n_layers * p->d_inner * p->d_conv, sizeof(float));
    s->ssm_state = calloc(p->n_layers * p->d_inner * p->d_state, sizeof(float));
    // ensure all mallocs went fine
    if (!s->input || !s->hidden_state || !s->xz || !s->x_db || !s->dt || 
        !s->dA || !s->dB || !s->temp || !s->y || !s->logits || 
        !s->conv_state || !s->ssm_state) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void reset_internal_state(Mamba* mamba) {
    // reset the internal state of the model
    RunState* s = &mamba->state;
    Config* p = &mamba->config;
    memset(s->conv_state, 0, p->n_layers * p->d_inner * p->d_conv * sizeof(float));
    memset(s->ssm_state, 0, p->n_layers * p->d_inner * p->d_state * sizeof(float));
}

void free_run_state(RunState* s) {
    free(s->input);
    free(s->hidden_state);
    free(s->xz);
    free(s->x_db);
    free(s->dt);
    free(s->dA);
    free(s->dB);
    free(s->temp);
    free(s->y);
    free(s->logits);
    free(s->conv_state);
    free(s->ssm_state);
}

void memory_map_weights(MambaWeights *w, Config* p, float* ptr) {
    unsigned long long n_layers = p->n_layers;
    
    // Debug helper function
    void debug_weight(const char* name, float* weight, int size) {
        printf("reading %s: first values: %f %f %f %f\n", 
               name, weight[0], weight[1], weight[2], weight[3]);
    }
    
    // get the pointers to the weights
    w->input_proj = ptr;
    debug_weight("input_proj", ptr, p->dim * p->input_size);
    ptr += p->dim * p->input_size;
    
    w->in_proj = ptr;
    debug_weight("in_proj", ptr, n_layers * (2 * p->d_inner) * p->dim);
    ptr += n_layers * (2 * p->d_inner) * p->dim;
    
    w->conv1d_weight = ptr;
    debug_weight("conv1d_weight", ptr, n_layers * p->d_inner * p->d_conv);
    ptr += n_layers * p->d_inner * 1 * p->d_conv;
    
    w->conv1d_bias = ptr;
    debug_weight("conv1d_bias", ptr, n_layers * p->d_inner);
    ptr += n_layers * p->d_inner;
    
    w->x_proj = ptr;
    debug_weight("x_proj", ptr, n_layers * (p->dt_rank + 2 * p->d_state) * p->d_inner);
    ptr += n_layers * (p->dt_rank + 2 * p->d_state) * p->d_inner;
    
    w->dt_proj_weight = ptr; 
    debug_weight("dt_proj_weight", ptr, n_layers * p->d_inner * p->dt_rank);
    ptr += n_layers * p->d_inner * p->dt_rank;
    
    w->dt_proj_bias = ptr;
    debug_weight("dt_proj_bias", ptr, n_layers * p->d_inner);
    ptr += n_layers * p->d_inner;
    
    w->A = ptr;
    debug_weight("A", ptr, n_layers * p->d_inner * p->d_state);
    ptr += n_layers * p->d_inner * p->d_state;
    
    w->D = ptr;
    debug_weight("D", ptr, n_layers * p->d_inner);
    ptr += n_layers * p->d_inner;
    
    w->out_proj = ptr;
    debug_weight("out_proj", ptr, n_layers * p->dim * p->d_inner);
    ptr += n_layers * p->dim * p->d_inner;
    
    w->norm = ptr;
    debug_weight("norm", ptr, n_layers * p->dim);
    ptr += n_layers * p->dim;
    
    w->final_norm = ptr;
    debug_weight("final_norm", ptr, p->dim);
    ptr += p->dim;
    
    w->classifier = ptr;
    debug_weight("classifier", ptr, p->num_classes * p->dim);
}

void load_model_file(char* model_path, Config* config, MambaWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(model_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", model_path); exit(EXIT_FAILURE); }
    // read the magic number
    unsigned int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic != 0x4d616d62) { fprintf(stderr, "Invalid magic number: %x\n", magic); exit(EXIT_FAILURE); }
    // read the version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 1) { fprintf(stderr, "Invalid version: %d\n", version); exit(EXIT_FAILURE); }
    // read the config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the model weights into the data pointer
    *fd = open(model_path, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + (256 / 4);
    memory_map_weights(weights, config, weights_ptr);
}

void load_model(Mamba* m, char* model_path) {
    // read the Config and the Weights from the model file
    load_model_file(model_path, &m->config, &m->weights, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state, &m->config);
}

void free_model(Mamba* m) {
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * weight[j] * ss;
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

void shift_matrix_left(float* matrix, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            matrix[i * cols + j] = matrix[i * cols + j + 1];
        }
    }
}

void update_last_column(float* matrix, float* x, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        matrix[i * cols + cols - 1] = x[i];
    }
}

void rowwise_dot_product(float* out, float* matrix, float* weights, int rows, int cols) {
    // matrix[rows,cols], weights[cols] -> out[rows]
    // this is a dot product of each row of the matrix with the weights
    // i.e. out[i] = matrix[i,:] @ weights
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float val = 0.0f;
        for (int j = 0; j < cols; j++) {
            val += matrix[i * cols + j] * weights[j];
        }
        out[i] = val;
    }
}

void matmul(float* xout, float* x, float* w, int d, int n) {
    // w[d,n] @ x[n] -> xout[d]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void linear(float* xout, float* x, float* w, float* b, int d, int n) {
    // w[d,n] @ x[n] + b[d] -> xout[d]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val + b[i];
    }
}

void broadcast_multiply(float* out, float* x, float* y, int d, int n) {
    // x[d], y[d,n] -> out[d,n]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            out[index] = x[i] * y[index];
            //out[i * n + j] = x[i] * y[i * n + j];
        }
    }
}

void elementwise_multiply(float* result, float* matrix1, float* matrix2, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        result[i] = matrix1[i] * matrix2[i];
    }
}

void elementwise_add(float* result, float* matrix1, float* matrix2, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        result[i] = matrix1[i] + matrix2[i];
    }
}

void elementwise_multiply_and_add(float* result, float* matrix1, float* matrix2, float* matrix3, int total_elements) {
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        result[i] = matrix1[i] * matrix2[i] + matrix3[i];
    }
}

void outer_product(float* out, float* x, float* y, int d, int n) {
    // x[d], y[n] -> out[d,n]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = x[i] * y[j];
        }
    }
}

void sum_along_last_dim(float* result, float* matrix, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float val = 0.0f;
        for (int j = 0; j < cols; j++) {
            val += matrix[i * cols + j];
        }
        result[i] = val;
    }
}

void forward_layer(Mamba* mamba, unsigned long long l, float* hidden_state) {
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    int dim = p->dim, d_inner = p->d_inner, d_conv = p->d_conv, d_state = p->d_state, dt_rank = p->dt_rank;
    float* dA = s->dA;  // (d_inner, d_state)
    float* dB = s->dB;  // (d_inner, d_state)
    float* y  = s->y;   // (d_inner)

    // conv_state, ssm_state = self._get_states_from_cache(inference_params)
    float* conv_state = s->conv_state + l * d_inner * d_conv;
    float* ssm_state  = s->ssm_state  + l * d_inner * d_state;

    // xz = self.in_proj(hidden_states)  # hidden_states: (dim), in_proj (2*d_inner, dim), xz (2*d_inner)
    matmul(s->xz, hidden_state, w->in_proj + l * 2*d_inner*dim, 2*d_inner, dim);
    // x, z = xz.chunk(2, dim=-1)
    float* x = s->xz;            // x (d_inner)
    float* z = s->xz + d_inner;  // z (d_inner)


    // Conv step

    // conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    shift_matrix_left(conv_state, d_inner, d_conv);
    // conv_state[:, -1] = x
    update_last_column(conv_state, x, d_inner, d_conv);
    // x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
    elementwise_multiply(s->temp, conv_state, w->conv1d_weight + l*d_inner*d_conv, d_inner * d_conv);
    sum_along_last_dim(x, s->temp, d_inner, d_conv);
    // x = x + self.conv1d.bias
    elementwise_add(x, x, w->conv1d_bias + l*d_inner, d_inner);
    // x = F.silu(x)
    for (int i = 0; i < d_inner; i++) {
        x[i] = silu(x[i]);
    }


    // SSM step

    // x_db = self.x_proj(x)   # x_db (dt_rank+2*d_state)
    matmul(s->x_db, x, w->x_proj + l*(dt_rank+2*d_state)*d_inner, dt_rank+2*d_state, d_inner);
    // dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    float *dt = s->x_db;                     // dt (dt_rank)
    float *B = s->x_db + dt_rank;            // B  (d_state)
    float *C = s->x_db + dt_rank + d_state;  // C  (d_state)

    // dt = self.dt_proj(dt)   # dt (dt_rank), dt_proj_weight (d_inner, dt_rank), dt_proj_bias (d_inner)
    linear(s->dt, dt, w->dt_proj_weight + l*d_inner*dt_rank, w->dt_proj_bias + l*d_inner, d_inner, dt_rank);
    dt = s->dt;  // NOTE: dt is now bigger: (d_inner) instead of (dt_rank)
    // dt = F.softplus(dt)
    for (int i = 0; i < d_inner; i++) {
        dt[i] = softplus(dt[i]);
    }

    //  Discretize A and B
    // dA = torch.exp(torch.einsum("d,dn->dn", dt, self.A))   # A (d_inner, d_state), dA (d_inner, d_state)
    broadcast_multiply(dA, dt, w->A + l*d_inner*d_state, d_inner, d_state);
    for (int i = 0; i < d_inner * d_state; i++) {
        dA[i] = expf(dA[i]);
    }
    // dB = torch.einsum("d,n->dn", dt, B)    # dt (d_inner), B (d_state), dB (d_inner, d_state)
    outer_product(dB, dt, B, d_inner, d_state);

    //  Update ssm_state
    // ssm_state.copy_(ssm_state * dA + rearrange(x, "d -> d 1") * dB)
    broadcast_multiply(s->temp, x, dB, d_inner, d_state);
    elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp, d_inner * d_state);

    //  Compute y
    // y = torch.einsum("dn,n->d", ssm_state, C) # ssm_state (d_inner, d_state), C (d_state), y (d_inner)
    rowwise_dot_product(y, ssm_state, C, d_inner, d_state);
    // y = y + self.D * x
    elementwise_multiply_and_add(y, w->D + l*d_inner, x, y, d_inner);
    // y = y * F.silu(z)  # (d_inner)
    for (int i = 0; i < d_inner; i++) {
        y[i] = y[i] * silu(z[i]);
    }

    // hidden_state = self.out_proj(y)  # out_proj (dim, d_inner), hidden_state (dim)
    matmul(hidden_state, y, w->out_proj + l*dim*d_inner, dim, d_inner);
}

float* forward(Mamba* mamba, float* input) {
    // convenience variables
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    int dim = p->dim;
    float *hidden_state = s->hidden_state;
    
    // Project input to model dimension
    matmul(hidden_state, input, w->input_proj, dim, p->input_size);
    memcpy(s->input, hidden_state, dim * sizeof(float));
    
    // forward all layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // normalize input
        rmsnorm(hidden_state, s->input, w->norm + l * dim, dim);
        // forward layer
        forward_layer(mamba, l, hidden_state);
        // residual connection
        for (int i = 0; i < dim; i++) {
            hidden_state[i] += s->input[i];
            s->input[i] = hidden_state[i];
        }
    }
    
    // final normalization
    rmsnorm(hidden_state, hidden_state, w->final_norm, dim);
    
    // classifier head
    matmul(s->logits, hidden_state, w->classifier, p->num_classes, dim);
    
    // apply softmax
    softmax(s->logits, p->num_classes);
    
    return s->logits;
}

// Add these structures after the existing ones, before main()
typedef struct {
    float* images;  // flattened array of images (n_images * 784)
    unsigned char* labels;  // array of labels (n_images)
    int n_images;
} MNISTDataset;

void load_mnist_file(const char* image_path, const char* label_path, MNISTDataset* dataset) {
    FILE *img_file = fopen(image_path, "rb");
    FILE *label_file = fopen(label_path, "rb");
    if (!img_file || !label_file) { 
        fprintf(stderr, "Failed to open MNIST files\n"); 
        exit(1); 
    }

    // Read header info
    int magic, n_images, n_rows, n_cols;
    fread(&magic, 4, 1, img_file);
    fread(&n_images, 4, 1, img_file);
    fread(&n_rows, 4, 1, img_file);
    fread(&n_cols, 4, 1, img_file);
    
    // Convert from big-endian
    n_images = ((n_images & 0xFF000000) >> 24) | ((n_images & 0x00FF0000) >> 8) |
               ((n_images & 0x0000FF00) << 8)  | ((n_images & 0x000000FF) << 24);

    // Skip label header
    fread(&magic, 4, 1, label_file);
    fread(&magic, 4, 1, label_file);

    // Allocate memory
    dataset->n_images = n_images;
    dataset->images = malloc(n_images * 784 * sizeof(float));
    dataset->labels = malloc(n_images * sizeof(unsigned char));

    // Read images and normalize
    unsigned char pixel;
    for (int i = 0; i < n_images * 784; i++) {
        fread(&pixel, 1, 1, img_file);
        dataset->images[i] = pixel / 255.0f;
    }

    // Read labels
    fread(dataset->labels, 1, n_images, label_file);

    fclose(img_file);
    fclose(label_file);
}

void free_mnist_dataset(MNISTDataset* dataset) {
    free(dataset->images);
    free(dataset->labels);
}

float evaluate_model(Mamba* mamba, MNISTDataset* dataset) {
    int correct = 0;
    
    // Process all images
    for (int i = 0; i < dataset->n_images; i++) {
        // Get prediction
        float* logits = forward(mamba, &dataset->images[i * 784]);
        
        // Find predicted class
        int predicted_class = 0;
        float max_prob = logits[0];
        for (int j = 1; j < mamba->config.num_classes; j++) {
            if (logits[j] > max_prob) {
                max_prob = logits[j];
                predicted_class = j;
            }
        }
        
        // Check if correct
        if (predicted_class == dataset->labels[i]) {
            correct++;
        }

        // Print progress every 1000 images
        if ((i + 1) % 1000 == 0) {
            fprintf(stderr, "Processed %d/%d images. Current accuracy: %.2f%%\n", 
                    i + 1, dataset->n_images, (100.0f * correct) / (i + 1));
        }

        // Reset internal state for next image
        reset_internal_state(mamba);
    }
    
    return (100.0f * correct) / dataset->n_images;
}

// Replace the existing main() with this version
int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  Single image: %s <model.bin> <input.bin>\n", argv[0]);
        fprintf(stderr, "  Full dataset: %s <model.bin> <mnist_images.bin> <mnist_labels.bin>\n", argv[0]);
        exit(1);
    }

    // Load model
    Mamba mamba;
    load_model(&mamba, argv[1]);
    fprintf(stderr, "Model loaded: input_size=%d, dim=%d, n_layers=%d\n",
            mamba.config.input_size, mamba.config.dim, mamba.config.n_layers);

    if (argc == 3) {
        // Single image mode
        float input[784];
        FILE* f = fopen(argv[2], "rb");
        if (!f) { 
            fprintf(stderr, "Failed to open input file %s\n", argv[2]); 
            exit(1); 
        }
        
        // Read raw float32 data directly
        size_t read = fread(input, sizeof(float), 784, f);
        if (read != 784) {
            fprintf(stderr, "Failed to read input, got %zu values, expected 784\n", read);
            exit(1);
        }
        fclose(f);

        // Run inference
        float* logits = forward(&mamba, input);

        // Find highest probability class
        int max_class = 0;
        float max_prob = logits[0];
        for (int i = 1; i < mamba.config.num_classes; i++) {
            if (logits[i] > max_prob) {
                max_prob = logits[i];
                max_class = i;
            }
        }
        
        printf("%d\n", max_class);
        
    } else {
        // Full dataset evaluation mode
        MNISTDataset dataset;
        load_mnist_file(argv[2], argv[3], &dataset);
        fprintf(stderr, "Dataset loaded: %d images\n", dataset.n_images);

        float accuracy = evaluate_model(&mamba, &dataset);
        printf("Final accuracy: %.2f%%\n", accuracy);

        free_mnist_dataset(&dataset);
    }

    free_model(&mamba);
    return 0;
}
