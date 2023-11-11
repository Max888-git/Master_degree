#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <string.h>
#include <pthread.h>
#include <sys/sysinfo.h>

#define N_INPUT 784
#define M 10
#define N 42000
#define MAX_IMAGES count_images_in_directory("data/")

#define LEARNING_RATE 0.01
#define MOMENTUM 0.9

// Структура для представлення параметрів нейронної мережі
typedef struct {
    float W1[M][N_INPUT];
    float b1[M][1];
    float W2[M][M];
    float b2[M][1];
    float X[N_INPUT]; // додано масив X
} NeuralNetwork;

// Структура для передачі параметрів у потік
typedef struct {
    NeuralNetwork* nn;
    int thread_id;
} ThreadParams;

// Структура для збереження градієнтів для оптимізації з моментумом
typedef struct {
    float dW1[M][N_INPUT];
    float db1[M][1];
    float dW2[M][M];
    float db2[M][1];
} Gradients;

// Функція для ініціалізації параметрів потоків
void init_thread_params(NeuralNetwork* nn, int thread_id, ThreadParams* params) {
    params->nn = nn;
    params->thread_id = thread_id;

    // Додаткові дії ініціалізації, якщо необхідно
}

// Функція, яку викликає кожен потік для обробки частини даних
void* thread_function(void* thread_params) {
    ThreadParams* params = (ThreadParams*)thread_params;

    // Отримати параметри потоку
    NeuralNetwork* nn = params->nn;
    int thread_id = params->thread_id;

    // Використовуємо nn і thread_id для обробки частини даних

    return NULL;
}

// Функція для ініціалізації параметрів нейронної мережі
void init_params(NeuralNetwork* nn) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N_INPUT; j++) {
            nn->W1[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
        nn->b1[i][0] = ((float)rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < M; j++) {
            nn->W2[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
        nn->b2[i][0] = ((float)rand() / RAND_MAX) - 0.5;
    }
}

float ReLU(float Z) {
    return (Z > 0) ? Z : 0;
}

void softmax(float* Z, int size) {
    float max_val = Z[0];
    for (int i = 1; i < size; i++) {
        if (Z[i] > max_val) {
            max_val = Z[i];
        }
    }

    float exp_sum = 0.0;
    for (int i = 0; i < size; i++) {
        Z[i] = expf(Z[i] - max_val);
        exp_sum += Z[i];
    }

    for (int i = 0; i < size; i++) {
        Z[i] /= exp_sum;
    }
}

void forward_prop(float W1[M][N_INPUT], float b1[M][1], float W2[M][M], float b2[M][1], float X[N_INPUT], float Z1[M], float A1[M], float Z2[M], float A2[M]) {
    for (int i = 0; i < M; i++) {
        Z1[i] = b1[i][0];
        for (int j = 0; j < N_INPUT; j++) {
            Z1[i] += W1[i][j] * X[j];
        }
        A1[i] = ReLU(Z1[i]);
    }

    for (int i = 0; i < M; i++) {
        Z2[i] = b2[i][0];
        for (int j = 0; j < M; j++) {
            Z2[i] += W2[i][j] * A1[j];
        }
    }

    softmax(Z2, M);
    for (int i = 0; i < M; i++) {
        A2[i] = Z2[i];
    }
}

float ReLU_deriv(float Z) {
    return (Z > 0) ? 1.0 : 0.0;
}

void one_hot(int* Y, int size, int num_classes, float* one_hot_Y) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < num_classes; j++) {
            one_hot_Y[i * num_classes + j] = (j == Y[i]) ? 1.0 : 0.0;
        }
    }
}

// Функція категоріальної крос-ентропії
float categorical_crossentropy(float* predicted_probs, float* true_probs, int num_classes) {
    float loss = 0.0;

    for (int i = 0; i < num_classes; i++) {
        loss += true_probs[i] * logf(predicted_probs[i] + 1e-10); // Додаємо невелику константу, щоб уникнути ділення на нуль
    }

    return -loss; // Мінус, оскільки оптимізація полягає у мінімізації втрат
}

// Функція для обчислення втрат для навчального прикладу
float compute_loss(NeuralNetwork* nn, float* X, int* Y) {
    float Z1[M], A1[M], Z2[M], A2[M];

    // Викликаємо функцію forward_prop, яку ви вже маєте в коді
    forward_prop(nn->W1, nn->b1, nn->W2, nn->b2, X, Z1, A1, Z2, A2);

    // Додаємо код для отримання one-hot представлення міток Y
    float one_hot_Y[M * 1];
    one_hot(Y, 1, M, one_hot_Y);

    // Обчислюємо втрати за допомогою категоріальної крос-ентропії
    float loss = categorical_crossentropy(A2, one_hot_Y, M);

    return loss;
}

// Оновлення ваг та зсувів за допомогою зворотного поширення помилок
void update_parameters(NeuralNetwork* nn, float* X, int* Y, float learning_rate) {
    // Forward propagation для отримання активацій на кожному шарі
    float Z1[M], A1[M], Z2[M], A2[M];
    forward_prop(nn->W1, nn->b1, nn->W2, nn->b2, X, Z1, A1, Z2, A2);

    // One-hot представлення міток Y
    float one_hot_Y[M * 1];
    one_hot(Y, 1, M, one_hot_Y);

    // Зворотнє поширення помилок
    float dZ2[M];
    for (int i = 0; i < M; i++) {
        dZ2[i] = A2[i] - one_hot_Y[i];
    }

    // Градієнти для оновлення параметрів другого шару
    float dW2[M][M], db2[M][1];
    for (int i = 0; i < M; i++) {
        db2[i][0] = dZ2[i];
        for (int j = 0; j < M; j++) {
            dW2[i][j] = dZ2[i] * A1[j];
        }
    }

    // Зворотнє поширення помилок для першого шару
    float dZ1[M];
    for (int i = 0; i < M; i++) {
        dZ1[i] = 0;
        for (int j = 0; j < M; j++) {
            dZ1[i] += dZ2[j] * nn->W2[j][i];
        }
        dZ1[i] *= ReLU_deriv(Z1[i]);
    }

    // Градієнти для оновлення параметрів першого шару
    float dW1[M][N_INPUT], db1[M][1];
    for (int i = 0; i < M; i++) {
        db1[i][0] = dZ1[i];
        for (int j = 0; j < N_INPUT; j++) {
            dW1[i][j] = dZ1[i] * X[j];
        }
    }

    // Оновлення ваг та зсувів за допомогою градієнтного спуску
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N_INPUT; j++) {
            nn->W1[i][j] -= learning_rate * dW1[i][j];
        }
        nn->b1[i][0] -= learning_rate * db1[i][0];

        for (int j = 0; j < M; j++) {
            nn->W2[i][j] -= learning_rate * dW2[i][j];
        }
        nn->b2[i][0] -= learning_rate * db2[i][0];
    }
}

// Функція для обчислення градієнтів зворотнього поширення помилки
void backward_prop(NeuralNetwork* nn, float* X, int* Y, Gradients* gradients) {
    float Z1[M], A1[M], Z2[M], A2[M];
    float dZ1[M], dA1[M], dZ2[M], dA2[M];
    float dW1[M][N_INPUT], db1[M][1], dW2[M][M], db2[M][1];  // Оголошення градієнтів

    // Викликаємо функцію forward_prop, яку ви вже маєте в коді
    forward_prop(nn->W1, nn->b1, nn->W2, nn->b2, X, Z1, A1, Z2, A2);

    // Додаємо код для отримання one-hot представлення міток Y
    float one_hot_Y[M * 1];
    one_hot(Y, 1, M, one_hot_Y);

    // Обчислюємо градієнти для останнього шару
    for (int i = 0; i < M; i++) {
        dZ2[i] = A2[i] - one_hot_Y[i];
        dW2[i][0] = dZ2[i] * A1[i];
        db2[i][0] = dZ2[i];
    }

    // Обчислюємо градієнти для прихованого шару
    for (int i = 0; i < M; i++) {
        dA1[i] = 0;
        for (int j = 0; j < M; j++) {
            dA1[i] += nn->W2[j][i] * dZ2[j];
        }
        dZ1[i] = dA1[i] * ReLU_deriv(Z1[i]);
        dW1[i][0] = dZ1[i] * X[0];
        db1[i][0] = dZ1[i];
    }

    // Записуємо градієнти у структуру
    for (int i = 0; i < M; i++) {
        gradients->dW1[i][0] = dW1[i][0];
        gradients->db1[i][0] = db1[i][0];
        gradients->dW2[i][0] = dW2[i][0];
        gradients->db2[i][0] = db2[i][0];
    }
}

// Функція для оновлення ваг та зсувів з оптимізацією моментуму
void update_parameters_with_momentum(NeuralNetwork* nn, Gradients* gradients, Gradients* velocities) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N_INPUT; j++) {
            velocities->dW1[i][j] = MOMENTUM * velocities->dW1[i][j] + LEARNING_RATE * gradients->dW1[i][j];
            nn->W1[i][j] -= velocities->dW1[i][j];
        }
        velocities->db1[i][0] = MOMENTUM * velocities->db1[i][0] + LEARNING_RATE * gradients->db1[i][0];
        nn->b1[i][0] -= velocities->db1[i][0];

        for (int j = 0; j < M; j++) {
            velocities->dW2[i][j] = MOMENTUM * velocities->dW2[i][j] + LEARNING_RATE * gradients->dW2[i][j];
            nn->W2[i][j] -= velocities->dW2[i][j];
        }
        velocities->db2[i][0] = MOMENTUM * velocities->db2[i][0] + LEARNING_RATE * gradients->db2[i][0];
        nn->b2[i][0] -= velocities->db2[i][0];
    }
}

// Функція для навчання нейронної мережі
void train_neural_network(NeuralNetwork* nn, float** X_train, int* Y_train, int num_samples, int num_epochs) {
    Gradients gradients;
    Gradients velocities;

    // Ініціалізуємо швидкість для алгоритму моментуму
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N_INPUT; j++) {
            velocities.dW1[i][j] = 0.0;
        }
        velocities.db1[i][0] = 0.0;
        for (int j = 0; j < M; j++) {
            velocities.dW2[i][j] = 0.0;
        }
        velocities.db2[i][0] = 0.0;
    }

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int sample = 0; sample < num_samples; sample++) {
            float* X = X_train[sample];
            int Y = Y_train[sample];

            compute_loss(nn, X, &Y);
            backward_prop(nn, X, &Y, &gradients);

            // Оновлюємо ваги та зсуви з моментумом
            update_parameters_with_momentum(nn, &gradients, &velocities);
        }
    }
}

// Функція для завантаження зображень
int loading_image(const char* filename, float X[N_INPUT]) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Помилка відкриття файлу %s\n", filename);
        return -1;
    }

    fseek(file, 16, SEEK_SET);  // Пропустити заголовок (16 байт)

    for (int i = 0; i < N_INPUT; i++) {
        unsigned char pixel;
        if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Помилка читання файлу %s\n", filename);
            fclose(file);
            return -1;
        }
        X[i] = (float)pixel / 255.0;
    }

    fclose(file);
    return 0;
}

int count_images_in_directory(const char* directory) {
    int count = 0;
    DIR* dir = opendir(directory);
    if (dir == NULL) {
        return 0;
    }

    struct dirent* entry;
    while ((entry = readdir(dir))) {
        if (entry->d_type == DT_REG) {
            count++;
        }
    }
    closedir(dir);

    return count;
}

float** allocate_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }
    return matrix;
}

void free_matrix(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int check_images_in_directory(const char* directory) {
    DIR* dir = opendir(directory);
    if (dir == NULL) {
        return 0;
    }
    closedir(dir);
    return 1;
}

// Додаємо функцію для парсингу XML
int parse_xml_annotation(const char* xml_filename) {
    FILE* file = fopen(xml_filename, "r");
    if (!file) {
        fprintf(stderr, "Помилка відкриття файлу %s\n", xml_filename);
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Пошук рядка <object>
        if (strstr(line, "<object>") != NULL) {
            char object_name[256];
            while (fgets(line, sizeof(line), file) && strstr(line, "</object>") == NULL) {
                // Пошук рядка <name>
                if (strstr(line, "<name>") != NULL) {
                    sscanf(line, "        <name>%[^<]</name>", object_name);
                    printf("Label: %s\n", object_name);
                }
            }
        }
    }

    fclose(file);
    return 0;
}

int main() {
    NeuralNetwork nn;
    init_params(&nn);

    const char* main_directory = "data";

    if (!check_images_in_directory(main_directory)) {
        printf("Каталог %s не містить зображень\n", main_directory);
        return 1;
    }

    int num_samples = 0;
    int max_samples = MAX_IMAGES;

    DIR* main_dir = opendir(main_directory);
    if (main_dir == NULL) {
        printf("Помилка відкриття каталогу %s\n", main_directory);
        return 1;
    }

    // Динамічно визначаємо кількість ядер
    int num_cores = get_nprocs();

    // Динамічно визначаємо кількість потоків (може бути менше кількості ядер)
    int num_threads = num_cores;

    // Створюємо масив ідентифікаторів потоків
    pthread_t threads[num_threads];

    // Ініціалізуємо параметри потоків
    ThreadParams thread_params[num_threads];
    for (int i = 0; i < num_threads; i++) {
    init_thread_params(&nn, i, &thread_params[i]);
    }

    struct dirent* main_entry;
    while ((main_entry = readdir(main_dir))) {
        if (main_entry->d_type == DT_DIR && strcmp(main_entry->d_name, ".") != 0 && strcmp(main_entry->d_name, "..") != 0) {
            const char* sub_directory = main_entry->d_name;
            char sub_directory_path[256];
            snprintf(sub_directory_path, sizeof(sub_directory_path), "%s/%s", main_directory, sub_directory);

            if (!check_images_in_directory(sub_directory_path)) {
                continue;
            }

            DIR* sub_dir = opendir(sub_directory_path);
            struct dirent* sub_entry;
            while ((sub_entry = readdir(sub_dir)) && num_samples < max_samples) {
                if (sub_entry->d_type == DT_REG && (strstr(sub_entry->d_name, ".png") != NULL)) {
                    char filename[256];
                    snprintf(filename, sizeof(filename), "%s/%s", sub_directory_path, sub_entry->d_name);
                    int result = loading_image(filename, nn.X);
                    if (result != 0) {
                        printf("Помилка при завантаженні зображення %s\n", filename);
                    } else {
                        // Отримайте мітку для зображення, використовуючи ваш XML-парсер
                        int xml_result = parse_xml_annotation("your_xml_filename.xml");
                        if (xml_result != 0) {
                            printf("Помилка при парсингу XML-файлу\n");
                        }

                        // Можемо використати nn.X та Y для обробки завантажених зображень
                        num_samples++;
                    }
                }
            }
            closedir(sub_dir);
        }
    }

    closedir(main_dir);

    // Створюємо потоки і запускаємо їх
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_function, (void*)&thread_params[i]);
    }

    // Очікуємо завершення потоків
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
