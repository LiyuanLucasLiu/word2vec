#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int negative = 0;
const int table_size = 1e8;
int *table;

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