/* Inference for Mamba model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>  // Add this include for uint32_t
#include <stdbool.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// Include LEO2 hardware accelerator headers
#include "leo2_libs.h"
#include "mannix_accelerator.h"
#include "mannix_main.h"
#include "mannix_regs_define.h"
#include "mannixlib.h"

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
    
    // LEO2 hardware accelerator structures
    Allocator_t allocator;
    MatAllocator_t mat_allocator;
    char fm_do_pad_allign;
    Matrix_t *hw_matrices; // Array to hold hardware matrices
    int hw_matrices_count;
    int hw_initialized;
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
        // printf("reading %s: first values: %f %f %f %f\n", 
        //        name, weight[0], weight[1], weight[2], weight[3]);
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
    
    // Initialize hardware accelerator structures
    m->hw_initialized = 0;
    initialize_hw_accelerator(m);
}

void free_model(Mamba* m) {
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

// Initialize LEO2 hardware accelerator for Mamba model
void initialize_hw_accelerator(Mamba* m) {
    if (m->hw_initialized) return;
    
    Config* p = &m->config;
    int dim = p->dim, d_inner = p->d_inner, dt_rank = p->dt_rank, d_state = p->d_state;
    int input_size = p->input_size, num_classes = p->num_classes;
    
    // Allocate memory for hardware matrices
    // We need matrices for:
    // 1. input_proj, in_proj, x_proj, dt_proj, out_proj, classifier
    // 2. Various intermediate results like xz, x_db, etc.
    int max_matrix_count = p->n_layers * 10 + 10; // Conservative estimate
    
    // Allocate memory for hw matrices array
    m->hw_matrices = (Matrix_t*)malloc(max_matrix_count * sizeof(Matrix_t));
    m->hw_matrices_count = 0;
    
    // Create shared memory allocator - size depends on our model parameters
    // This is a conservative estimate of memory needed
    size_t mem_size = p->n_layers * (
        dim * input_size +           // input_proj
        2 * d_inner * dim +          // in_proj
        (dt_rank + 2 * d_state) * d_inner + // x_proj
        d_inner * dt_rank +          // dt_proj_weight
        dim * d_inner +              // out_proj
        num_classes * dim            // classifier
    ) * sizeof(float) + 1024 * 1024; // Extra buffer

    // Allocate memory
    char* data = (char*)malloc(mem_size);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for hardware accelerator\n");
        exit(EXIT_FAILURE);
    }
    
    // Create allocator
    createAllocator(&m->allocator, data, mem_size);
    
    // Create matrix allocator
    createMatrixAllocator(&m->mat_allocator, m->hw_matrices, max_matrix_count);
    
    // Set padding alignment for memory
    m->fm_do_pad_allign = 1;
    
    m->hw_initialized = 1;
    
    printf("LEO2 hardware accelerator initialized\n");
}

// Create a hardware compatible matrix from host float array with int8 quantization
Matrix_t* create_hw_matrix(Mamba* m, int rows, int cols, float* data) {
    Matrix_t* matrix = &m->hw_matrices[m->hw_matrices_count++];
    creatMatrix(rows, cols, matrix, &m->allocator, m->fm_do_pad_allign);
    
    // Copy data from host array to hardware memory
    // Since LEO2 only supports int8, we need to quantize float to int8
    for (int i = 0; i < rows * cols; i++) {
        if (data) {
            // Quantize float to int8 range (-127 to 127)
            float scaled_val = data[i] * 127.0f;
            // Clamp to int8 range to avoid overflow
            if (scaled_val > 127.0f) scaled_val = 127.0f;
            if (scaled_val < -127.0f) scaled_val = -127.0f;
            // Convert to int8
            char val = (char)scaled_val;
            put_byte_over_apb((volatile char*)(matrix->data + i), val, 0);
        } else {
            // Zero initialization
            put_byte_over_apb((volatile char*)(matrix->data + i), 0, 0);
        }
    }
    
    return matrix;
}

// Free hardware accelerator resources
void free_hw_accelerator(Mamba* m) {
    if (!m->hw_initialized) return;
    
    // Free allocated memory
    if (m->allocator.data) {
        free(m->allocator.data);
        m->allocator.data = NULL;
    }
    
    // Free hw_matrices array
    if (m->hw_matrices) {
        free(m->hw_matrices);
        m->hw_matrices = NULL;
    }
    
    m->hw_matrices_count = 0;
    m->hw_initialized = 0;
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
        // Set last column to zero after shifting
        matrix[i * cols + cols - 1] = 0.0f;
    }
}

void initialize_conv_state(float* conv_state, int d_inner, int d_conv) {
    // Initialize with zeros for padding=d_conv-1 on the left side
    #pragma omp parallel for
    for (int i = 0; i < d_inner; i++) {
        for (int j = 0; j < d_conv; j++) {
            conv_state[i * d_conv + j] = 0.0f;
        }
    }
}

void update_last_column(float* matrix, float* x, int rows, int cols) {
    // Update the last column with new input values
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        matrix[i * cols + (cols - 1)] = x[i];
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

// Hardware-accelerated matrix multiplication
void hw_matmul(Mamba* mamba, float* xout, float* x, float* w, int d, int n) {
    if (!mamba->hw_initialized) {
        // Fallback to CPU implementation if hardware not initialized
        matmul(xout, x, w, d, n);
        return;
    }
    
    // Create matrices for LEO2 hardware accelerator
    // Input matrix (n x 1)
    Matrix_t in_matrix;
    creatMatrix(n, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
    
    // Weight matrix (d x n)
    Matrix_t weight_matrix;
    creatMatrix(d, n, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
    
    // Output matrix (d x 1)
    Matrix_t out_matrix;
    creatMatrix(d, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
    
    // Create dummy bias vector (required by hardware API but not used)
    Matrix_t dummy_bias;
    creatMatrix(d, 1, &dummy_bias, &mamba->allocator, mamba->fm_do_pad_allign);
    
    // Convert float input to hardware format and copy to in_matrix
    for (int i = 0; i < n; i++) {
        // Convert float to int8/char for hardware
        char val = (char)(x[i] * 127.0f);
        put_byte_over_apb((volatile char*)(in_matrix.data + i), val, 0);
    }
    
    // Convert float weights to hardware format and copy to weight_matrix
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            // Convert float to int8/char for hardware
            char val = (char)(w[i * n + j] * 127.0f);
            put_byte_over_apb((volatile char*)(weight_matrix.data + i * n + j), val, 0);
        }
    }
    
    // Initialize bias with zeros
    for (int i = 0; i < d; i++) {
        put_byte_over_apb((volatile char*)(dummy_bias.data + i), 0, 0);
    }
    
    // Call hardware accelerator for matrix multiplication
    xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &dummy_bias, &out_matrix, 0, mamba->fm_do_pad_allign);
    
    // Convert hardware result back to float and copy to xout
    for (int i = 0; i < d; i++) {
        // Read output from hardware memory and convert back to float
        char val = get_byte_over_apb((volatile char*)(out_matrix.data + i), 0);
        xout[i] = val / 127.0f;  // Convert from int8 to float
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
    // Using hardware acceleration for matrix multiplication
    if (mamba->hw_initialized) {
        // Create hardware matrices
        Matrix_t in_matrix;
        Matrix_t weight_matrix;
        Matrix_t out_matrix;
        Matrix_t dummy_bias;
        
        // Setup input matrix (dim x 1)
        creatMatrix(dim, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup weight matrix (2*d_inner x dim)
        creatMatrix(2*d_inner, dim, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup output matrix (2*d_inner x 1)
        creatMatrix(2*d_inner, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup dummy bias (required by hardware but not used)
        creatMatrix(2*d_inner, 1, &dummy_bias, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Copy data to hardware memory
        for (int i = 0; i < dim; i++) {
            char val = (char)(hidden_state[i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)in_matrix.data)[i]), val, 0);
        }
        
        for (int i = 0; i < 2*d_inner; i++) {
            for (int j = 0; j < dim; j++) {
                char val = (char)(w->in_proj[l * 2*d_inner*dim + i*dim + j] * 127.0f);
                put_byte_over_apb(&(((volatile char *)weight_matrix.data)[i*dim + j]), val, 0);
            }
            // Initialize bias with zeros
            put_byte_over_apb(&(((volatile char *)dummy_bias.data)[i]), 0, 0);
        }
        
        // Call hardware accelerator
        xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &dummy_bias, &out_matrix, 0, mamba->fm_do_pad_allign);
        
        // Copy results back
        for (int i = 0; i < 2*d_inner; i++) {
            char val = get_byte_over_apb(&(((volatile char *)out_matrix.data)[i]), 0);
            s->xz[i] = val / 127.0f;
        }
    } else {
        // Fallback to CPU implementation
        matmul(s->xz, hidden_state, w->in_proj + l * 2*d_inner*dim, 2*d_inner, dim);
    }
    
    // x, z = xz.chunk(2, dim=-1)
    float* x = s->xz;            // x (d_inner)
    float* z = s->xz + d_inner;  // z (d_inner)


    // Conv step
    // In PyTorch, conv1d weight shape is (d_inner, 1, d_conv) for groups=d_inner
    // Each channel convolves with its own d_conv filter
    shift_matrix_left(conv_state, d_inner, d_conv);
    update_last_column(conv_state, x, d_inner, d_conv);
    
    // Depthwise convolution - each channel uses its own filter
    // Keep on CPU as it's not efficiently mapped to LEO2 FC accelerator
    #pragma omp parallel for
    for (int i = 0; i < d_inner; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d_conv; j++) {
            sum += conv_state[i * d_conv + j] * w->conv1d_weight[l*d_inner*d_conv + i*d_conv + j];
        }
        x[i] = sum + w->conv1d_bias[l*d_inner + i];  // Add bias per channel
        x[i] = x[i] * sigmoid(x[i]);  // SiLU activation: x * sigmoid(x)
    }


    // SSM step

    // x_db = self.x_proj(x)   # x_db (dt_rank+2*d_state)
    if (mamba->hw_initialized) {
        // Create hardware matrices
        Matrix_t in_matrix;
        Matrix_t weight_matrix;
        Matrix_t out_matrix;
        Matrix_t dummy_bias;
        
        // Setup input matrix (d_inner x 1)
        creatMatrix(d_inner, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup weight matrix (dt_rank+2*d_state x d_inner)
        creatMatrix(dt_rank+2*d_state, d_inner, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup output matrix (dt_rank+2*d_state x 1)
        creatMatrix(dt_rank+2*d_state, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup dummy bias
        creatMatrix(dt_rank+2*d_state, 1, &dummy_bias, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Copy data to hardware memory
        for (int i = 0; i < d_inner; i++) {
            char val = (char)(x[i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)in_matrix.data)[i]), val, 0);
        }
        
        for (int i = 0; i < dt_rank+2*d_state; i++) {
            for (int j = 0; j < d_inner; j++) {
                char val = (char)(w->x_proj[l*(dt_rank+2*d_state)*d_inner + i*d_inner + j] * 127.0f);
                put_byte_over_apb(&(((volatile char *)weight_matrix.data)[i*d_inner + j]), val, 0);
            }
            // Initialize bias with zeros
            put_byte_over_apb(&(((volatile char *)dummy_bias.data)[i]), 0, 0);
        }
        
        // Call hardware accelerator
        xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &dummy_bias, &out_matrix, 0, mamba->fm_do_pad_allign);
        
        // Copy results back
        for (int i = 0; i < dt_rank+2*d_state; i++) {
            char val = get_byte_over_apb(&(((volatile char *)out_matrix.data)[i]), 0);
            s->x_db[i] = val / 127.0f;
        }
    } else {
        // Fallback to CPU implementation
        matmul(s->x_db, x, w->x_proj + l*(dt_rank+2*d_state)*d_inner, dt_rank+2*d_state, d_inner);
    }
    
    // dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    float *dt = s->x_db;                     // dt (dt_rank)
    float *B = s->x_db + dt_rank;            // B  (d_state)
    float *C = s->x_db + dt_rank + d_state;  // C  (d_state)

    // dt = self.dt_proj(dt)   # dt (dt_rank), dt_proj_weight (d_inner, dt_rank), dt_proj_bias (d_inner)
    if (mamba->hw_initialized) {
        // Create hardware matrices
        Matrix_t in_matrix;
        Matrix_t weight_matrix;
        Matrix_t bias_matrix;
        Matrix_t out_matrix;
        
        // Setup input matrix (dt_rank x 1)
        creatMatrix(dt_rank, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup weight matrix (d_inner x dt_rank)
        creatMatrix(d_inner, dt_rank, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup bias matrix (d_inner x 1)
        creatMatrix(d_inner, 1, &bias_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup output matrix (d_inner x 1)
        creatMatrix(d_inner, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Copy data to hardware memory
        for (int i = 0; i < dt_rank; i++) {
            char val = (char)(dt[i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)in_matrix.data)[i]), val, 0);
        }
        
        for (int i = 0; i < d_inner; i++) {
            for (int j = 0; j < dt_rank; j++) {
                char val = (char)(w->dt_proj_weight[l*d_inner*dt_rank + i*dt_rank + j] * 127.0f);
                put_byte_over_apb(&(((volatile char *)weight_matrix.data)[i*dt_rank + j]), val, 0);
            }
            // Copy bias values
            char bias_val = (char)(w->dt_proj_bias[l*d_inner + i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)bias_matrix.data)[i]), bias_val, 0);
        }
        
        // Call hardware accelerator
        xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &bias_matrix, &out_matrix, 0, mamba->fm_do_pad_allign);
        
        // Copy results back
        for (int i = 0; i < d_inner; i++) {
            char val = get_byte_over_apb(&(((volatile char *)out_matrix.data)[i]), 0);
            s->dt[i] = val / 127.0f;
        }
    } else {
        // Fallback to CPU implementation
        linear(s->dt, dt, w->dt_proj_weight + l*d_inner*dt_rank, w->dt_proj_bias + l*d_inner, d_inner, dt_rank);
    }
    
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

    // hidden_state = self.out_proj(y)  # out_proj (dim, d_inner), hidden_state (dim)˜
    if (mamba->hw_initialized) {
        // Create hardware matrices
        Matrix_t in_matrix;
        Matrix_t weight_matrix;
        Matrix_t out_matrix;
        Matrix_t dummy_bias;
        
        // Setup input matrix (d_inner x 1)
        creatMatrix(d_inner, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup weight matrix (dim x d_inner)
        creatMatrix(dim, d_inner, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup output matrix (dim x 1)
        creatMatrix(dim, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup dummy bias
        creatMatrix(dim, 1, &dummy_bias, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Copy data to hardware memory
        for (int i = 0; i < d_inner; i++) {
            char val = (char)(y[i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)in_matrix.data)[i]), val, 0);
        }
        
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < d_inner; j++) {
                char val = (char)(w->out_proj[l*dim*d_inner + i*d_inner + j] * 127.0f);
                put_byte_over_apb(&(((volatile char *)weight_matrix.data)[i*d_inner + j]), val, 0);
            }
            // Initialize bias with zeros
            put_byte_over_apb(&(((volatile char *)dummy_bias.data)[i]), 0, 0);
        }
        
        // Call hardware accelerator
        xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &dummy_bias, &out_matrix, 0, mamba->fm_do_pad_allign);
        
        // Copy results back
        for (int i = 0; i < dim; i++) {
            char val = get_byte_over_apb(&(((volatile char *)out_matrix.data)[i]), 0);
            hidden_state[i] = val / 127.0f;
        }
    } else {
        // Fallback to CPU implementation
        matmul(hidden_state, y, w->out_proj + l*dim*d_inner, dim, d_inner);
    }
}

float* forward(Mamba* mamba, float* input) {
    // convenience variables
    Config* p = &mamba->config;
    MambaWeights* w = &mamba->weights;
    RunState* s = &mamba->state;
    int dim = p->dim;
    float *hidden_state = s->hidden_state;
    
    // Project input to model dimension using hardware acceleration
    if (mamba->hw_initialized) {
        // Create hardware matrices
        Matrix_t in_matrix;
        Matrix_t weight_matrix;
        Matrix_t out_matrix;
        Matrix_t dummy_bias;
        
        // Setup input matrix (input_size x 1)
        creatMatrix(p->input_size, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup weight matrix (dim x input_size)
        creatMatrix(dim, p->input_size, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup output matrix (dim x 1)
        creatMatrix(dim, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup dummy bias
        creatMatrix(dim, 1, &dummy_bias, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Copy input data to hardware memory
        for (int i = 0; i < p->input_size; i++) {
            char val = (char)(input[i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)in_matrix.data)[i]), val, 0);
        }
        
        // Copy weight data to hardware memory
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < p->input_size; j++) {
                char val = (char)(w->input_proj[i * p->input_size + j] * 127.0f);
                put_byte_over_apb(&(((volatile char *)weight_matrix.data)[i * p->input_size + j]), val, 0);
            }
            // Initialize bias with zeros
            put_byte_over_apb(&(((volatile char *)dummy_bias.data)[i]), 0, 0);
        }
        
        // Call hardware accelerator
        xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &dummy_bias, &out_matrix, 0, mamba->fm_do_pad_allign);
        
        // Copy results back to hidden_state
        for (int i = 0; i < dim; i++) {
            char val = get_byte_over_apb(&(((volatile char *)out_matrix.data)[i]), 0);
            hidden_state[i] = val / 127.0f;
        }
    } else {
        // Fallback to CPU implementation
        matmul(hidden_state, input, w->input_proj, dim, p->input_size);
    }
    
    memcpy(s->input, hidden_state, dim * sizeof(float));
    
    // Forward all layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // Normalize input
        rmsnorm(hidden_state, s->input, w->norm + l * dim, dim);
        // Forward layer (already hardware-accelerated)
        forward_layer(mamba, l, hidden_state);
        // Residual connection
        for (int i = 0; i < dim; i++) {
            hidden_state[i] += s->input[i];
            s->input[i] = hidden_state[i];
        }
    }
    
    // Final normalization
    rmsnorm(hidden_state, hidden_state, w->final_norm, dim);
    
    // Classifier head using hardware acceleration
    if (mamba->hw_initialized) {
        // Create hardware matrices
        Matrix_t in_matrix;
        Matrix_t weight_matrix;
        Matrix_t out_matrix;
        Matrix_t dummy_bias;
        
        // Setup input matrix (dim x 1)
        creatMatrix(dim, 1, &in_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup weight matrix (num_classes x dim)
        creatMatrix(p->num_classes, dim, &weight_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup output matrix (num_classes x 1)
        creatMatrix(p->num_classes, 1, &out_matrix, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Setup dummy bias
        creatMatrix(p->num_classes, 1, &dummy_bias, &mamba->allocator, mamba->fm_do_pad_allign);
        
        // Copy input data to hardware memory
        for (int i = 0; i < dim; i++) {
            char val = (char)(hidden_state[i] * 127.0f);
            put_byte_over_apb(&(((volatile char *)in_matrix.data)[i]), val, 0);
        }
        
        // Copy classifier weights to hardware memory
        for (int i = 0; i < p->num_classes; i++) {
            for (int j = 0; j < dim; j++) {
                char val = (char)(w->classifier[i * dim + j] * 127.0f);
                put_byte_over_apb(&(((volatile char *)weight_matrix.data)[i * dim + j]), val, 0);
            }
            // Initialize bias with zeros
            put_byte_over_apb(&(((volatile char *)dummy_bias.data)[i]), 0, 0);
        }
        
        // Call hardware accelerator
        xlrtr_fc_with_activate(&in_matrix, &weight_matrix, &dummy_bias, &out_matrix, 0, mamba->fm_do_pad_allign);
        
        // Copy results back to logits
        for (int i = 0; i < p->num_classes; i++) {
            char val = get_byte_over_apb(&(((volatile char *)out_matrix.data)[i]), 0);
            s->logits[i] = val / 127.0f;
        }
    } else {
        // Fallback to CPU implementation
        matmul(s->logits, hidden_state, w->classifier, p->num_classes, dim);
    }
    
    // Apply softmax (keep on CPU as it's not a matrix multiplication)
    softmax(s->logits, p->num_classes);
    
    return s->logits;
}


int main(int argc, char *argv[]) {
    bool quiet_mode = false;
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quiet") == 0) {
            quiet_mode = true;
        }
    }

    // if (argc != 3) {
    //     fprintf(stderr, "Usage: %s <model.bin> <input.bin>\n", argv[0]);
    //     fprintf(stderr, "Input file should contain 784 float32 values for MNIST image\n");
    //     exit(1);
    // }

    // Load model
    Mamba mamba;
    load_model(&mamba, argv[1]);
    if (quiet_mode) {
        // don't print anything
    } else {
    fprintf(stderr, "Model loaded: input_size=%d, dim=%d, n_layers=%d\n",
            mamba.config.input_size, mamba.config.dim, mamba.config.n_layers);
    }
    // Load input image
    float input[784];
    FILE* f = fopen(argv[2], "rb");
    if (!f) { 
        fprintf(stderr, "Failed to open input file %s\n", argv[2]); 
        exit(1); 
    }
    
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
    
    if (quiet_mode) {
        printf("%d\n", max_class);
    } else {
        printf("Predicted class: %d with probability: %f\n", max_class, max_prob);
    }
    
    free_model(&mamba);
    return 0;
}
