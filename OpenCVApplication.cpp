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
			for (int filterI = 0; filterI < offsetI.size; filterI++) {
				for (int filterJ = 0; filterJ < offsetJ.size; filterJ++) {
					int newI = i + offsetI[filterI];
					int newJ = j + offsetJ[filterJ];
					if (isInRange(newI, newJ)) {
						newValue += filter[filterI][filterJ] * src.at<uchar>(newI, newJ);
					}
				}
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

Mat gradientAndDirection(Mat src, Mat* grad, Mat* dir) {
	*grad = src.clone();
	*dir = src.clone();
	//paralelizare
	std::vector< std::vector<double>> filterJ =
	{ { -1, 0, 1 },
	{ -1, 0, 1 },
	{ -1, 0, 1 } };
	std::vector< std::vector<double>> filterI =
	{ { 1, 1, 1 },
	{ 0, 0, 0 },
	{ -1, -1, -1 } };
	std::vector<double> offsetI = { -1, 0, 1 };
	std::vector<double> offsetJ = { -1, 0, 1 };

	//paralelizare
	Mat convI = convolution(src, filterI, offsetI, offsetJ);
	Mat convJ = convolution(src, filterJ, offsetI, offsetJ);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			(*grad).at<uchar>(i, j) = sqrt(pow(convI.at<uchar>(i, j), 2) + pow(convJ.at<uchar>(i, j), 2));
			(*dir).at<uchar>(i,j) = arct
		}
	}
}

Mat detectiePuncteMuchie(Mat src) {
	Mat dst = src.clone();
	Mat noiseFiltration = gaussFiltration(src);
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
	Mat filter = gaussFiltration(src);
	//Mat contours = detectiePuncteMuchie(src);
	//Mat inverse = ipm(contours);

	imshow("input image", src);
	//imshow("countouring", contours);
	//imshow("inverse image", inverse);
	imshow("Filtration", filter);
	waitKey();
		
	return 0;
}