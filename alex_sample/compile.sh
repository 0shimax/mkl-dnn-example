g++ -std=c++11 -fopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/build/src alex_training.cpp -lmkldnn -o ./alex_training
