#include "../include/quotients.cuh"
#include <cstdio>

typedef struct {
    secure_field_point point;
    uint32_t *columns;
    qm31 *values;
    uint32_t size;
} column_sample_batch;

void column_sample_batches_for(
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        const uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        column_sample_batch *result
) {
    unsigned int offset = 0;
    for (unsigned int index = 0; index < sample_size; index++) {
        result[index].point = sample_points[index];
        result[index].columns = &sample_column_indexes[offset];
        result[index].values = &sample_column_values[offset];
        result[index].size = sample_column_and_values_sizes[index];
        offset += sample_column_and_values_sizes[index];
    }
};

void accumulate_quotients(
        point domain_initial_point,
        point domain_step,
        uint32_t domain_size,
        uint32_t **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3
) {
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
    printf("%d\n", domain_size);
    printf("%d %d %d %d\n", sample_points[0].x.a.a, sample_points[0].x.a.b, sample_points[0].x.b.a, sample_points[0].x.b.b);
    printf("%d %d %d %d\n", sample_points[0].y.a.a, sample_points[0].y.a.b, sample_points[0].y.b.a, sample_points[0].y.b.b);
    printf("%d %d\n", sample_column_indexes[0], sample_column_indexes[1]);
    printf("%d\n", sample_column_and_values_sizes[0]);
    printf("%d %d %d %d\n", sample_column_values[0].a.a, sample_column_values[0].a.b, sample_column_values[0].b.a, sample_column_values[0].b.b);
    printf("%d %d %d %d\n", sample_column_values[1].a.a, sample_column_values[1].a.b, sample_column_values[1].b.a, sample_column_values[1].b.b);
    printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");

    auto *column_sample_batches = (column_sample_batch*) malloc(sizeof(column_sample_batch) * sample_size);
    column_sample_batches_for(
            sample_points,
            sample_column_indexes,
            sample_column_values,
            sample_column_and_values_sizes,
            sample_size,
            column_sample_batches
    );

    printf("%d %d %d %d\n", column_sample_batches[0].point.x.a.a, column_sample_batches[0].point.x.a.b, column_sample_batches[0].point.x.b.a, column_sample_batches[0].point.x.b.b);
    printf("%d %d %d %d\n", column_sample_batches[0].point.y.a.a, column_sample_batches[0].point.y.a.b, column_sample_batches[0].point.y.b.a, column_sample_batches[0].point.y.b.b);
}
