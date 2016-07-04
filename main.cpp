// Distributed two-dimensional Discrete FFT transform
// W B Cooper
// Project 1
//


#include <iostream>
#include <math.h>
#include "Complex.h"
#include "InputImage.h"
#include <pthread.h>
using namespace std;

bool DEBUG = true; // to print out the data after each operation if true
bool idft = true;  // perform idft after dft if true
string finalOutput_dft = "WBCooperResult.txt";
string finalOutput_idft= "WBCooperResult_idft.txt";
Complex* data;
int width, height;
//int N; // width*height of data

// Barrier parameters
int nThreads = 16;
int P0; // Total threads for Barrier functions
int count;    // Number of threads in the barrier.
int threadCounter();
pthread_mutex_t countMutex,exitMutex;
pthread_cond_t exitCond;
pthread_mutex_t startCountMutex;
int threadCount;
bool* localSense;
bool globalSense;


void Transpose( Complex* data, int width);
void WriteImageData(const char* newFileName, Complex* data, int w, int h);
void debug(string filename,Complex* data, int width, bool inv_flag);
void* Transform2D(void* v);
void Transform1D(Complex* h, int N, bool idft);
void MyBarrier(unsigned myId);
void MyBarrier_Init();

int main(int argc, char** argv)
{

// data input section

	string fileName("Tower.txt");
    // See if number of thread specified on command lind
    if (argc > 1) nThreads = atol(argv[1]);
    // See if file name specified on command line
    if (argc > 2) fileName = string(argv[2]);
    InputImage image(fileName.c_str());

    data = image.GetImageData();
    width = image.GetWidth();
    height = image.GetHeight();

    int th;
  //  pthread_t threads;
    pthread_t threads[nThreads];

    // ======================= START FORWARD DFT ========================= //
    cout<<"Starting Forward DFT.."<<endl;
    idft = false;

    if (DEBUG) debug("000_original_array", data, width, false);

    MyBarrier_Init();
    pthread_mutex_init(&exitMutex,0);
    pthread_cond_init(&exitCond,0);
    pthread_mutex_init(&startCountMutex,0);

   //  Step 1: n threads perform row transformation. Each thread works on width*height/nThreads rows
    for(th = 0; th < nThreads;th++)
    {
        pthread_create(&threads[th], 0, Transform2D, (void *)th);
    }
    MyBarrier(nThreads);

	if (DEBUG) debug("001_after1D",data, width, false);
	Transpose(data, width);
	if (DEBUG) debug("002_after_transpose_before2D",data, width, false);

    MyBarrier_Init();
    pthread_mutex_init(&exitMutex,0);
    pthread_cond_init(&exitCond,0);
    pthread_mutex_init(&startCountMutex,0);

    for(th = 0; th < nThreads;th++)
    {
        pthread_create(&threads[th], 0, Transform2D, (void *)th);
    }
    MyBarrier(nThreads);

    if (DEBUG) debug("003_after2D", data, width, false);
	Transpose(data, width);
    if (DEBUG) debug("004_after_transpose_after2D", data, width, false);
	    WriteImageData(finalOutput_dft.c_str(), data, width, height); //save dft transform

    // ======================= START INVERSE DFT ========================= //
    cout<<"Starting Inverse DFT.."<<endl;
    idft = true;
    if (DEBUG) debug("000_original_array", data, width, true);
    for(th = 0; th < nThreads;th++)
    {
       pthread_create(&threads[th], 0, Transform2D, (void *)th);
    }
    MyBarrier(nThreads);

    if (DEBUG) debug("001_after1D",data, width, true);
    Transpose(data, width);								//transpose matrix
    if (DEBUG) debug("002_after_transpose_before2D",data, width, true);
    for(th = 0; th < nThreads;th++)
    {
       pthread_create(&threads[th], 0, Transform2D, (void *)th);
    }
    MyBarrier(nThreads);

    if (DEBUG) debug("003_after2D", data, width, true);
    Transpose(data, width);							//transpose matrix
    if (DEBUG) debug("004_after_transpose_after2D", data, width, true);
    WriteImageData(finalOutput_idft.c_str(), data, width, height);
	return(0);
}

void MyBarrier_Init()// you will likely need some parameters)
{
  P0 = nThreads + 1; // 16 threads + main thread
  count = nThreads + 1;
  pthread_mutex_init(&countMutex, 0); //Initialize Mutex used for counting threads
  // Create and initialize the localSense array, 1 entry per thread
  localSense = new bool[P0];
  for(int i = 0; i < P0; ++i)
  {
    localSense[i] = true;
  }
  globalSense = true; // initialize global sense.
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(unsigned myId) // Again likely need parameters
{
    localSense[myId] = !localSense[myId]; // Toggle private thread variable
    if(threadCounter() == 1)
    {
      // All threads here, reset count and toggle globalSense
      count = P0;
      globalSense = localSense[myId];
    }
    else
    {
      while(globalSense != localSense[myId]) {  } //Spin
    }
}

int threadCounter()
{
  //cout<<"Inside threadCounter()"<<endl;
  pthread_mutex_lock(&countMutex);
	int myCount = count;
	count--;
	pthread_mutex_unlock(&countMutex);
	return myCount;
}

void Transpose( Complex* a, int width)
{
    int height = width;
    Complex c [width*height];

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			c[j*width+i] = a[i*width+j];
		}
	}
	for (int i = 0; i < width*height; i++) a[i] = c[i];
}
void WriteImageData(const char* newFileName, Complex* d,int w, int h)
{
  ofstream ofs(newFileName);
  if (!ofs)
    {
      cout << "Can't create output image " << newFileName << endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c].Mag() << " ";
        }
      ofs << endl;
    }
}

void debug(string filename,Complex* array1, int width, bool inv_flag)
{
    string prefix;
    if (inv_flag == true) prefix = "inv_"; else  prefix = "";
    filename =  prefix + filename;
	WriteImageData(filename.append(".wbctxt").c_str(),array1,width,width);
	cout << "Printing intermediate result in File: "<<filename<< endl;
}
//******************************************************************************
// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.


unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = width; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value

  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

void* Transform2D(void* v)
{
    unsigned long threadId = (unsigned long)v;
    const int rowsPerThread = width/nThreads;
	const int startRow = rowsPerThread*threadId;
	const int endRow = (rowsPerThread*(threadId+1)-1);
    for (int row = startRow; row <= endRow; row++)
    {
        Transform1D(&data[row*width], width, idft);
    }
 //   cout << "threadId: " << threadId <<endl;
    MyBarrier(threadId);
    return(0);
}

void Transform1D(Complex* h, int N, bool idft)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)

 double invDFT = -1;
 if (idft == true) invDFT = 1;
 Complex* x = new Complex[N];

 unsigned v;
 for (int i = 0; i < N; ++i)
 {
   v = i;
   x[ReverseBits(v)] = h[i];
 }
 for (int i = 1; i <=log2(N); ++i){
   int Num_blocks = N/pow(2,i);
   for (int j=0; j<Num_blocks;++j){
    Complex W(0.0,0.0);
    // int k_w =0
    int n_w = N/Num_blocks;
    Complex* temp = new Complex[n_w] ;
    for (int k = 0; k < n_w; ++k){
    Complex W = Complex((double)cos(2*M_PI*k/n_w),(invDFT)*sin(2*M_PI*k/n_w));
      if (k < n_w/2)
         temp[k]  = x[j*n_w+k] + W*x[j*n_w+k+n_w/2];
      else
        temp[k]  = x[j*n_w+k-n_w/2] + (W*x[j*n_w+k]);
    }
    for (int u = 0; u < n_w; u++)
    {
      x[j*n_w + u] = temp[u];
    }
    delete temp;
   }
 }
  for (int i = 0; i < N; i++) {
   if (idft == false) h[i] = x[i]; else h[i] = x[i]/Complex(width,0);
  }

}
//
