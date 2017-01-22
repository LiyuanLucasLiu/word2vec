#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define FREE(x) if (x != NULL) {free(x);}
#define CHECKNULL(x) if (x == NULL) {printf("Memory allocation failed\n"); exit(1);}
#define NRAND next_random = next_random * (unsigned long long)25214903917 + 11;

typedef float real;                    // Precision of float numbers

struct supervision {
  long long function_id;
  long long label;
};

struct training_ins {
  long long id;
  long long c_num;
  long long *cList;
  long long sup_num;
  struct supervision *supList;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// struct vocab_word *vocab;
long long  *cCount;
int binary = 0, debug_mode = 2, reSample = 10, min_count = 5, num_threads = 1, min_reduce = 1;
long long c_size = 0, c_length = 100, l_size = 1, l_length = 1000, d_size, NONE_idx, tot_c_count = 0;
real lambda1 = 0.3, lambda2 = 0.3;
long long ins_num = 191965, ins_count_actual = 0;
long long iters = 10;
long print_every = 1000;
real alpha = 0.025, starting_alpha, sample = 1e-4;

struct training_ins * data;
real *c, *l, *d, *cneg, *db, *lb;
real *o;
real *ph1, ph2;
real *sigTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;
long long *table;

oid InitNet() {
  long long a, b;
  a = posix_memalign((void **)&c, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(c)
  a = posix_memalign((void **)&cneg, 128, (long long)c_size * c_length * sizeof(real));
  CHECKNULL(cneg)
  a = posix_memalign((void **)&l, 128, (long long)l_size * l_length * sizeof(real));
  CHECKNULL(l)
  a = posix_memalign((void **)&d, 128, (long long)d_size * l_length * sizeof(real));
  CHECKNULL(d)
  a = posix_memalign((void **)&o, 128, (long long)c_length * l_length * sizeof(real));
  CHECKNULL(o)
  a = posix_memalign((void **)&lb, 128, (long long)l_size * sizeof(real));
  CHECKNULL(lb)
  a = posix_memalign((void **)&db, 128, (long long)d_size * sizeof(real));
  CHECKNULL(db)
  cCount = (long long *) calloc(c_size,  sizeof(long long));
  CHECKNULL(cCount)
  memset(cCount, 0, c_size); //ini1
  ph1 = (real*) calloc(d_size, sizeof(real));
  CHECKNULL(ph1)
  ph2 = (real*) calloc(d_size, sizeof(real));
  CHECKNULL(ph2)

  for (b = 0; b < c_size; b++) for (a = 0; a < c_length; a++) {
    c[b * c_length + a] = (rand() / (real)RAND_MAX - 0.5) / c_length;
    cneg[b * c_length + a] = 0;
  }
  for (b = 0; b < l_size; b++) lb[b] = 0;
  for (b = 0; b < d_size; b++) db[b] = 0;
  for (b = 0; b < d_size; b++) ph1[b] = 0.5;
  for (b = 0; b < d_size; b++) ph2[b] = 0.5;
  for (b = 0; b < l_size; b++) for (a = 0; a < l_length; a++)
    l[b * l_length + a] = 0;//(rand() / (real)RAND_MAX - 0.5) / l_length;
  for (b = 0; b < d_size; b++) for (a = 0; a < l_length; a++)
    d[b * l_length + a] = 0;//(rand() / (real)RAND_MAX - 0.5) / d_length; 
  for (b = 0; b < l_length; b++) for (a = 0; a < c_length; a++)
    o[b * c_length + a] = 0; 
}