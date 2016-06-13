// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "IPM.h"
#define _USE_MATH_DEFINES
#include <math.h>
using namespace cv;
using namespace std;

#define THRESHOLD 200
int height, width;

bool isInRange(int i, int j);

Mat openImage() {
	char fname[MAX_PATH] = "C:\\Users\\Oana\\Desktop\\licenta\\srcgray.png";
	//while (openFileDlg(fname))
	//{
	//openFileDlg(fname);
	Mat src, graySrc;
	src = imread(fname);

	if (src.channels() == 3)
		cvtColor(src, graySrc, CV_BGR2GRAY);
	else
		src.copyTo(graySrc);

	width = graySrc.cols;
	height = graySrc.rows;
	//}
	return graySrc;
}

Mat convolution(Mat src, std::vector< std::vector<double>> filter, std::vector<double> offsetI, std::vector<double> offsetJ) {
	Mat dst = src.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double newValue = 0;
			for (int filterI = 0; filterI < offsetI.size(); filterI++) {
				for (int filterJ = 0; filterJ < offsetJ.size(); filterJ++) {
					int newI = i + offsetI[filterI];
					int newJ = j + offsetJ[filterJ];
					if (isInRange(newI, newJ)) {
						double value = src.at<uchar>(newI, newJ);
						newValue =newValue + filter[filterI][filterJ] * value;
					}
				}
			}
			if (newValue > 255) {
				newValue = 255;
			}
			if (newValue < 0) {
				newValue = 0;
			}
			dst.at<uchar>(i, j) = newValue;
		}
	}
	return dst;
}

Mat gaussFiltration(Mat src) {
	std::vector< std::vector<double>> filter = { { 1.0 / 16, 1.0 / 8, 1.0 / 16 }, { 1.0 / 8, 1.0 / 4, 1.0 / 8 }, { 1.0 / 16, 1.0 / 8, 1.0 / 16 } };
	std::vector<double> offsetI = { -1, 0, 1 };
	std::vector<double> offsetJ = { -1, 0, 1 };
	return convolution(src, filter, offsetI, offsetJ);
}

void gradientModuleAndDirection(Mat src, Mat* mod, Mat* dir) {
	*mod = src.clone();
	*dir = src.clone();
	//paralelizare
	std::vector< std::vector<double>> filterX = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	std::vector< std::vector<double>> filterY = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
	std::vector<double> offsetI = { -1, 0, 1 };
	std::vector<double> offsetJ = { -1, 0, 1 };

	//paralelizare
	Mat convX = convolution(src, filterX, offsetI, offsetJ);
	Mat convY = convolution(src, filterY, offsetI, offsetJ);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			(*mod).at<uchar>(i, j) = sqrt(pow(convX.at<uchar>(i, j), 2) + pow(convY.at<uchar>(i, j), 2));
			(*dir).at<uchar>(i, j) = std::atan2(convY.at<uchar>(i, j), convX.at<uchar>(i, j)) * 180 / M_PI;
		}
	}
}

int dirCuantification(double value) {
	double halfPeriod = 45.0 / 2;
	double raport = value / halfPeriod;
	if ((raport >= -1 && raport < 1) || (raport >= 7 && raport <= 8) || (raport >= -8 && raport < -7)) {
		return 2;
	}
	if ((raport >= 1 && raport < 3) || (raport >= -7 && raport < -5)) {
		return 1;
	}
	if ((raport >= 3 && raport < 5) || (raport >= -5 && raport < -3)) {
		return 0;
	}
	if ((raport >= 5 && raport < 7) || (raport >= -3 && raport < -1)) {
		return 3;
	}
	return -1;
}

Mat nonMaxSuprimation(Mat module, Mat dir) {
	// dest has to be a module clone
	Mat dest = module.clone();
	std::vector<double> offsetI = { 1, -1, 0, 1 };
	std::vector<double> offsetJ = { 0, 1, 1, 1 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int cuantifiedDirection = (dirCuantification(dir.at<uchar>(i, j)));
			if (cuantifiedDirection == -1) {
				printf("Error ! dir = %d\n", dir.at<uchar>(i, j));
				break;
			}
			if (module.at<uchar>(i, j) != 0) {
				int newI1 = i + offsetI[cuantifiedDirection];
				int newJ1 = j + offsetJ[cuantifiedDirection];
				int newI2 = i - offsetI[cuantifiedDirection];
				int newJ2 = j - offsetJ[cuantifiedDirection];
				if (isInRange(newI1, newJ1)) {
					if (module.at<uchar>(i, j) < module.at<uchar>(newI1, newJ1)) {
						dest.at<uchar>(i, j) = 0;
					}
					if (module.at<uchar>(i, j) == module.at<uchar>(newI1, newJ1)){
						if (dest.at<uchar>(newI1, newJ1) != 0) {
						//if destination is already 0 for neighbour => keep current pixel
						//if not 0, make current pixel 0
							dest.at<uchar>(i, j) = 0;
						}
					}
				}
				if (isInRange(newI2, newJ2)) {
					if (module.at<uchar>(i, j) < module.at<uchar>(newI2, newJ2)) {
						dest.at<uchar>(i, j) = 0;
					}
					if (module.at<uchar>(i, j) == module.at<uchar>(newI2, newJ2)){
						if (dest.at<uchar>(newI2, newJ2) != 0) {
						//if destination already 0 for neighbour => keep current pixel
						//if not 0, make current pixel 0
						dest.at<uchar>(i, j) = 0;
						}
					}
				}
			}
		}
	}
	return dest;
}

int binarizationThreshold(Mat module) {
	const double p = 0.01;
	int *hist = new int[256]();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int normalizedValue = module.at<uchar>(i, j) / (4 * sqrt(2));
			hist[normalizedValue]++;
		}
	}
	int nrNonMuchii = (1 - p) * (width * height - hist[0]);
	int adaptiveThreshold = 0;
	int sum = 0;
	for (int i = 1; i <= 255; i++) {
		sum += hist[i];
		if (sum > nrNonMuchii) {
			adaptiveThreshold = i;
			break;
		}
	}
	return adaptiveThreshold;
}

struct point{
	int i;
	int j;
};

Mat histereza(Mat mod, int adaptiveThreshold) {
	Mat dst = mod.clone();
	const double k = 0.4;
	const int muchieTare = 255;
	const int muchieSlaba = 128;
	double lowThreshold = k * adaptiveThreshold;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (mod.at<uchar>(i, j) >= adaptiveThreshold) {
				dst.at<uchar>(i, j) = muchieTare;
			}else if (mod.at<uchar>(i, j) >= lowThreshold) {
				dst.at<uchar>(i, j) = muchieSlaba;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	std::deque<point*> points;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (dst.at<uchar>(i, j) == muchieTare) {
				point *p = new point();
				p->i = i;
				p->j = j;
				points.push_back(p);
			}
		}
	}
	//find neighbours
	std::vector<double> offsetI = { -1, 0, 1 };
	std::vector<double> offsetJ = { -1, 0, 1 };
	while (!points.empty()) {
		point *p = points.front();
		points.pop_front();
		for (int i = 0; i < offsetI.size(); i++) {
			for (int j = 0; j < offsetJ.size(); j++) {
				int newI = p->i + offsetI[i];
				int newJ = p->j + offsetJ[j];
				if (isInRange(newI, newJ)) {
					if (dst.at<uchar>(newI, newJ) == muchieSlaba) {
						dst.at<uchar>(newI, newJ) = muchieTare;
						points.push_back(p);
					}
				}
			}
		}
	}
	//eliminate non-muchii
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (dst.at<uchar>(i, j) == muchieSlaba) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

Mat binarizarePrinHistereza(Mat src) {
	int adaptiveThreshold = binarizationThreshold(src);
	return histereza(src, adaptiveThreshold);
}

Mat detectiePuncteMuchie(Mat src) {
	Mat noiseFiltration = gaussFiltration(src);
	Mat mod, dir;
	gradientModuleAndDirection(noiseFiltration, &mod, &dir);
	Mat suprimated = nonMaxSuprimation(mod, dir);
	imshow("suprimated", suprimated);
	Mat dst = binarizarePrinHistereza(suprimated);
	return dst;
}

bool isInRange(int i, int j) {
	if (i < 0) {
		return false;
	}
	if (i >= height) {
		return false;
	}
	if (j < 0) {
		return false;
	}
	if (j >= width) {
		return false;
	}
	return true;
}

int searchForLeft(Mat src, int lastLeftJ, int i) {
	int currentJ = lastLeftJ;
	while (src.at<uchar>(i, currentJ) > THRESHOLD) {
		currentJ++;
	}
	return currentJ;
}

int searchForRight(Mat src, int lastRightJ, int i) {
	int currentJ = lastRightJ;
	while (src.at<uchar>(i, currentJ) > THRESHOLD) {
		currentJ--;
	}
	return currentJ;
}

Mat ipm(Mat src) {

	Mat dst = src.clone();

	//start for bottom
	// start j from middle
	int middle = width / 2;
	int leftJ = middle, rightJ = middle;
	int i = height - 1;
	while (src.at<uchar>(i, leftJ) < THRESHOLD && leftJ > 0) {
		leftJ--;
	}
	while (src.at<uchar>(i, rightJ) < THRESHOLD && rightJ < width - 1) {
		rightJ++;
	}
	dst.at<uchar>(i, leftJ) = 255;
	dst.at<uchar>(i, rightJ) = 255;
	int dLeft = 0;
	int dRight = 0;
	for (int i = height - 2; i >= height/2; i--) {
		int newLeftJ = searchForLeft(src, leftJ, i);
		dLeft = dLeft + newLeftJ - leftJ;
		leftJ = newLeftJ;
		int newRightJ = searchForRight(src, rightJ, i);
		dRight = dRight + rightJ - newRightJ;
		rightJ = newRightJ;

		//moveLeft
		for (int j = 0; j <= leftJ - dLeft; j++) {
			dst.at<uchar>(i, j) = dst.at<uchar>(i, j + dLeft);
		}
		for (int j = leftJ - dLeft + 1; j <= leftJ; j++) {
			dst.at<uchar>(i, j) = 0;
		}

		//moveRight
		for (int j = width - 1; j >= rightJ + dRight; j--) {
			dst.at<uchar>(i, j) = dst.at<uchar>(i, j - dRight);
		}
		for (int j = rightJ + dRight - 1; j >= rightJ; j--) {
			dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}

int main()
{
	system("cls");
	destroyAllWindows();
	Mat src = openImage();
	Mat contours = detectiePuncteMuchie(src);
	//Mat inverse = ipm(contours);

	imshow("input image", src);
	imshow("contouring", contours);
	//imshow("inverse image", inverse);
	waitKey();
		
	return 0;
}