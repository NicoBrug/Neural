#include "../../includes/loader/mnist.h"
#include <iostream>

#include <fstream>
#include <assert.h>

using namespace Eigen;


mnist::mnist(std::string image_file,
			   std::string label_file,
			   int num) :
  m_size(0),
  m_rows(0),
  m_cols(0)
{
  load_images(image_file, num);
  load_labels(label_file, num);
}

mnist::mnist(std::string image_file,
			   std::string label_file) :
  mnist(image_file, label_file, 0)
{
  // empty
}

mnist::~mnist()
{
  // empty
}

int
mnist::to_int(char* p)
{
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
         ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}

void
mnist::load_images(std::string image_file, int num)
{
  std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
  char p[4];

  ifs.read(p, 4);
  int magic_number = to_int(p);
  //assert(magic_number == 0x803);

  ifs.read(p, 4);
  m_size = to_int(p);
  // limit
  if (num != 0 && num < m_size) m_size = num;

  ifs.read(p, 4);
  m_rows = to_int(p);

  ifs.read(p, 4);
  m_cols = to_int(p);

  char* q = new char[m_rows * m_cols];

  MatrixXd matrix(m_size,m_rows*m_cols);
  std::cout << "size" << m_size << std::endl;

  for (int i=0; i<m_size; ++i) {
    ifs.read(q, m_rows * m_cols);
    std::vector<double> image(m_rows * m_cols);
    for (int j=0; j<m_rows * m_cols; ++j) {
      image[j] = static_cast<unsigned char>(q[j]) / 255.0;
    }

    m_images.push_back(image);
    matrix.row(i) = Map<Matrix<double,1,784> >(image.data());

  }
  this->data.images = matrix;
  delete[] q;

  ifs.close();
}

void
mnist::load_labels(std::string label_file, int num)
{
  std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
  char p[4];

  ifs.read(p, 4);
  int magic_number = to_int(p);
  //assert(magic_number == 0x801);

  ifs.read(p, 4);
  int size = to_int(p);
  // limit


  if (num != 0 && num <= m_size){
      size = num;
  } 
  
  MatrixXd mnist_matrix(size,10);


  for (int i=0; i<size; ++i) {
    ifs.read(p, 1);
    int label = p[0];
    m_labels.push_back(label);
    
    MatrixXd labelArray(1,10);

     for (int j(0); j<10; j++){
            if (label == j){
                labelArray(0,j) = 1.0;
            }
            else{
                labelArray(0,j) = 0.0;
            }
      }
    mnist_matrix.row(i) = labelArray;
  }
  this->data.labels = mnist_matrix;

  ifs.close();
}
