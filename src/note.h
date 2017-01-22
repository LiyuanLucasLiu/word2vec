
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define FREE(x) if (x != NULL) {free(x);}
#define CHECKNULL(x) if (x == NULL) {printf("Memory allocation failed\n"); exit(1);}

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
long long c_size = 0, c_length = 100, l_size = 1, l_length = 1000, d_size;
long long ins_num = 191965, ins_count_actual = 0;
long long iters = 10;
real alpha = 0.025, starting_alpha, sample = 0;

struct training_ins * data;
real *c, *l, *d, *cneg, *db, *lb;
real *o;
real *expTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;
long long *table;

void InitUnigramTable()
void ReadWord(char *word, FILE *fin)
int GetWordHash(char *word)
int SearchVocab(char *word)
int ReadWordIndex(FILE *fin)
int AddWordToVocab(char *word)
int VocabCompare(const void *a, const void *b)
void DestroyVocab()
void SortVocab()
void ReduceVocab()
void LearnVocabFromTrainFile()
void SaveVocab()
void ReadVocab()
void InitNet()
void DestroyNet()
void *TrainModelThread(void *id)
void TrainModel()
int ArgPos(char *str, int argc, char **argv)