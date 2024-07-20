#ifndef FIELDS_H
#define FIELDS_H

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

typedef struct {
    uint32_t a;
    uint32_t b;
} cm31;

typedef struct {
    cm31 a;
    cm31 b;
} qm31;

#endif // FIELDS_H