// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "IPM.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <queue>
using namespace cv;
using namespace std;

int height, width;

bool isInRange(int i, int j);
int dirCuantification(double value);
//Mat borderTracing(Mat src);
void stretch(Mat src, Point pLeft, Point pRight, Point pIntersection);

Mat openImage(char* fname) {
	//char fname[MAX_PATH] = "C:\\Users\\Oana\\Desktop\\licenta\\srcgray.png";
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

	Mat dst = Mat(height, width, DataType<int>::type);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<int>(i, j) = (int)graySrc.at<uchar>(i, j);
		}
	}
	//}
	return dst;
}

Mat convolution(Mat src, std::vector< std::vector<double>> filter, std::vector<double> offsetI, std::vector<double> offsetJ) {
	Mat dst = Mat(height, width, DataType<int>::type);
	int ignoreSize = filter.size() / 2;
	for (int i = ignoreSize; i < height - ignoreSize; i++) {
		for (int j = ignoreSize; j < width - ignoreSize; j++) {
			double newValue = 0;
			for (int indexIFilter = 0; indexIFilter < offsetI.size(); indexIFilter++) {
				for (int indexJFilter = 0; indexJFilter < offsetJ.size(); indexJFilter++) {
					int newI = i + offsetI[indexIFilter];
					int newJ = j + offsetJ[indexJFilter];
					if (isInRange(newI, newJ)) {
						double value = src.at<int>(newI, newJ);
						newValue = newValue + filter[indexIFilter][indexJFilter] * value;
					}
				}
			}
	//		if (newValue > 255) {
		//		newValue = 255;
			//}
			//if (newValue < 0) {
				//newValue = 0;
			//}
			dst.at<int>(i, j) = newValue;
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
	std::vector< std::vector<double>> filterX = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	std::vector< std::vector<double>> filterY = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
	std::vector<double> offsetI = { -1, 0, 1 };
	std::vector<double> offsetJ = { -1, 0, 1 };

	//paralelizare
	Mat convX = convolution(src, filterX, offsetI, offsetJ);
	Mat convY = convolution(src, filterY, offsetI, offsetJ);

	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			double moduleValue = sqrt(pow(convX.at<int>(i, j), 2.0) + pow(convY.at<int>(i, j), 2.0));
			(*mod).at<int>(i, j) = moduleValue;
			double dirValue = std::atan2(convY.at<int>(i, j), convX.at<int>(i, j)) * 180.0 / M_PI;
			(*dir).at<int>(i, j) = dirCuantification(dirValue);
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
	Mat dst = module.clone();
	std::vector<double> offsetI = { 1, -1, 0, 1 };
	std::vector<double> offsetJ = { 0, 1, 1, 1 };
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			int cuantifiedDirection = dir.at<int>(i, j);
			if (cuantifiedDirection == -1) {
				printf("Error ! dir = %d\n", dir.at<int>(i, j));
				break;
			}
			if (module.at<int>(i, j) != 0) {
				int newI1 = i + offsetI[cuantifiedDirection];
				int newJ1 = j + offsetJ[cuantifiedDirection];
				int newI2 = i - offsetI[cuantifiedDirection];
				int newJ2 = j - offsetJ[cuantifiedDirection];
				if (isInRange(newI1, newJ1)) {
					if (module.at<int>(i, j) <= module.at<int>(newI1, newJ1)) {
						dst.at<int>(i, j) = 0;
					}
				}
				if (isInRange(newI2, newJ2)) {
					if (module.at<int>(i, j) <= module.at<int>(newI2, newJ2)) {
						dst.at<int>(i, j) = 0;
					}
				}
			}
		}
	}
	return dst;
}

int binarizationThreshold(Mat module) {
	const double p = 0.01;
	int *hist = new int[256]();
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			int normalizedValue = module.at<int>(i, j) / (4 * sqrt(2));
			hist[normalizedValue]++;
		}
	}
	int nrNonMuchii = (1 - p) * ((width-2) * (height-2) - hist[0]);
	int adaptiveThreshold = 0;
	int sum = 0;
	for (int i = 1; i <= 255; i++) {
		sum += hist[i];
		if (sum >= nrNonMuchii) {
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
	const double k = 0.7;
	const int muchieTare = 255;
	const int muchieSlaba = 128;
	const int nonMuchie = 0;
	double lowThreshold = k * adaptiveThreshold;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (i == 0 || j == 0 || i == height - 1 || j == width - 1) {
				dst.at<int>(i, j) = nonMuchie;
			} else if (mod.at<int>(i, j) >= adaptiveThreshold) {
				dst.at<int>(i, j) = muchieTare;
			} else if (mod.at<int>(i, j) >= lowThreshold) {
				dst.at<int>(i, j) = muchieSlaba;
			} else {
				dst.at<int>(i, j) = nonMuchie;
			}
		}
	}
	std::vector<double> offsetI = { -1, 0, 1 };
	std::vector<double> offsetJ = { -1, 0, 1 };
	std::deque<point*> points;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (dst.at<int>(i, j) == muchieTare) {
				point *p = new point();
				p->i = i;
				p->j = j;
				points.push_back(p);
				while (!points.empty()) {
					point *newP = points.front();
					points.pop_front();
					for (int i = 0; i < offsetI.size(); i++) {
						for (int j = 0; j < offsetJ.size(); j++) {
							int newI = newP->i + offsetI[i];
							int newJ = newP->j + offsetJ[j];
							if (isInRange(newI, newJ)) {
								if (dst.at<int>(newI, newJ) == muchieSlaba) {
									dst.at<int>(newI, newJ) = muchieTare;
									points.push_back(newP);
								}
							}
						}
					}
				}
			}
		}
	}
	
	
	//eliminate non-muchii
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (dst.at<int>(i, j) == muchieSlaba) {
				dst.at<int>(i, j) = nonMuchie;
			}
		}
	}
	return dst;
}

Mat binarizarePrinHistereza(Mat src) {
	int adaptiveThreshold = binarizationThreshold(src);
	return histereza(src, adaptiveThreshold);
}

Mat canny(Mat src) {
	Mat noiseFiltration = gaussFiltration(src);
	Mat mod, dir;
	gradientModuleAndDirection(noiseFiltration, &mod, &dir);
	//imshow("Gradient modules", mod);
	Mat suprimated = nonMaxSuprimation(mod, dir);
	//imshow("Suprimarea non-maximelor", suprimated);
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

Mat showIntMat(String message, Mat src) {
	Mat dst = Mat(height, width, DataType<uchar>::type);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = (uchar)src.at<int>(i, j);
		}
	}
	//imshow(message, dst);
	return dst;
}

void computeHough(int x, int y, double *H) {
	double diag = sqrt(height * height + width * width);
	for (int teta = 0; teta < 360; teta++){
		int ro = x * cos(teta * M_PI / 180.0) + y * sin(teta * M_PI / 180.0);
		if (ro > 0 && ro < diag){
			H[ro * 360 + teta]++;
		}
	}
}

boolean isInLeftInterval(double theta) {
	double theta_grades = theta * 180 / M_PI;
	if (theta_grades < 60 && theta_grades > 30) {
		return true;
	}
	return false;
}

boolean isInRightInterval(double theta) {
	double theta_grades = theta * 180 / M_PI;
	if (theta_grades >120 && theta_grades < 150) {
		return true;
	}
	return false;
}

vector<Point> drawFoundLines(Vec2f lineExt, Vec2f lineInt, Mat img, boolean isLeft) {
	Point pt1Int, pt2Int, pt1Ext, pt2Ext;
	double rhoInt = lineInt[0];
	double rhoExt = lineExt[0];
	double aInt = cos(lineInt[1]), bInt = sin(lineInt[1]);
	double aExt = cos(lineExt[1]), bExt = sin(lineExt[1]);
	double x0Int = aInt*rhoInt, y0Int = bInt*rhoInt;
	pt1Int.x = cvRound(x0Int + 1000 * (-bInt));
	pt1Int.y = cvRound(y0Int + 1000 * (aInt));
	pt2Int.x = cvRound(x0Int - 1000 * (-bInt));
	pt2Int.y = cvRound(y0Int - 1000 * (aInt));

	double x0Ext = aExt*rhoExt, y0Ext = bExt*rhoExt;
	pt1Ext.x = cvRound(x0Ext + 1000 * (-bExt));
	pt1Ext.y = cvRound(y0Ext + 1000 * (aExt));
	pt2Ext.x = cvRound(x0Ext - 1000 * (-bExt));
	pt2Ext.y = cvRound(y0Ext - 1000 * (aExt));

	Point p1, p2;
	p1.x = pt1Int.x + (pt1Ext.x - pt1Int.x) / 2;
	p1.y = pt1Int.y + (pt1Ext.y - pt1Int.y) / 2;
	p2.x = pt2Int.x + (pt2Ext.x - pt2Int.x) / 2;
	p2.y = pt2Int.y + (pt2Ext.y - pt2Int.y) / 2;

	vector<Point> points;
	if (isLeft) {
		points.push_back(pt1Int);
		points.push_back(pt1Ext);
	}
	else {
		points.push_back(pt2Int);
		points.push_back(pt2Ext);
	}
	points.push_back(p1);
	points.push_back(p2);
	//polylines(img, points, true, Scalar(255, 255, 255), 2, CV_AA);
	//line(img, pt1Ext, pt2Ext, Scalar(255, 255, 255), 2, CV_AA);
	return points;
}

vector<double> equationParam(Point p1, Point p2) {
	double a, b, c;
	a = p1.y - p2.y;
	b = p2.x - p1.x;
	c = p1.x * p2.y - p2.x * p1.y;
	vector<double> parameters;
	parameters.push_back(a);
	parameters.push_back(b);
	parameters.push_back(c);
	return parameters;
}

Point findIntersectionPoint(Point p1Line1, Point p2Line1, Point p1Line2, Point p2Line2) {
	vector<double> paramLeft, paramRight;
	paramLeft = equationParam(p1Line1, p2Line1);
	paramRight = equationParam(p1Line2, p2Line2);
	double al = paramLeft[0], bl = paramLeft[1], cl = paramLeft[2];
	double ar = paramRight[0], br = paramRight[1], cr = paramRight[2];

	Point p;
	p.y = (ar * cl - al * cr) / (al * br - ar * bl);
	p.x = (-cl - bl * p.y) / al;
	return p;
}

void normalHough(Mat mat, Mat src) {
	vector<Vec2f> lines;
	//empiric
	HoughLines(mat, lines, 1, M_PI / 180, 100, 0, 0);
	Vec2f leftLineExt, leftLineInt, rightLineInt, rightLineExt;
	leftLineExt[1] = M_PI / 2;
	leftLineInt[1] = 0;
	rightLineExt[1] = M_PI / 2;
	rightLineInt[1] = M_PI;
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		if (isInLeftInterval(theta) || isInRightInterval(theta)) {
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(mat, pt1, pt2, Scalar(255, 255, 255), 1, LINE_AA);
			if (isInLeftInterval(theta)) {
				if (theta < leftLineExt[1]) {
					leftLineExt = lines[i];
				}
				if (theta > leftLineInt[1]) {
					leftLineInt = lines[i];
				}
			}
			if (isInRightInterval(theta)) {
				if (theta < rightLineInt[1]) {
					rightLineInt = lines[i];
				}
				if (theta > rightLineExt[1]) {
					rightLineExt = lines[i];
				}
			}
		}
	}
	vector<Point> pointsLeft, pointsRight;
	Point p1Left, p2Left, p1Right, p2Right;
	pointsLeft = drawFoundLines(leftLineExt, leftLineInt, src, true);
	pointsRight = drawFoundLines(rightLineExt, rightLineInt, src, false);

	Point intersectionPoint = findIntersectionPoint(pointsLeft[2], pointsLeft[3], pointsRight[2], pointsRight[3]);
	
	vector<Point> pointsLeftToDraw{pointsLeft[0], pointsLeft[1], intersectionPoint};
	//polylines(src, pointsLeftToDraw, true, Scalar(255, 255, 255), 2, CV_AA);

	vector<Point> pointsRightToDraw{pointsRight[0], pointsRight[1], intersectionPoint};
	//polylines(src, pointsRightToDraw, true, Scalar(255, 255, 255), 2, CV_AA);

	imshow("hough", mat);
	//imshow("detection", src);
	stretch(src, pointsLeft[0], pointsRight[0], intersectionPoint);
}

void stretch(Mat src, Point pLeft, Point pRight, Point pIntersection) {
	double yDif;
	if (pLeft.y > height) {
		yDif = abs(height - pIntersection.y);
	}
	else {
		yDif = abs(pLeft.y - pIntersection.y);
	}
	Point p1New, p2New;
	p1New.y = pIntersection.y + 2;//+ yDif / 5;
	p2New.y = pIntersection.y + 2; //yDif / 5;
	p1New.x = 0;
	p2New.x = width - 1;
	Point pIntersectionLeft = findIntersectionPoint(pLeft, pIntersection, p1New, p2New);
	Point pIntersectionRight = findIntersectionPoint(pRight, pIntersection, p1New, p2New);
	vector<Point> pointsToStretch {pLeft, pIntersectionLeft, pIntersectionRight, pRight};
	Mat aux = src.clone();
	polylines(aux, pointsToStretch, true, Scalar(255, 255, 255), 2, CV_AA);
	imshow("trapez", aux);

	Mat dst = src.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = 0;
		}
	}


	int yMin = min(pIntersectionLeft.y, pIntersectionRight.y);
	int yMax = max(pLeft.y, pRight.y);
	yMax = min(yMax, height - 1);
	for (int y = yMin; y <= yMax; y++) {
		double xl = findIntersectionPoint(pLeft, pIntersectionLeft, Point{ 0, y }, Point{ width - 1, y }).x;
		double xr = findIntersectionPoint(pRight, pIntersectionRight, Point{ 0, y }, Point{ width - 1, y }).x;
		for (int x = 0; x < width; x++) {
			int correspondingX = x * (xr - xl) / (width - 1) + xl;
			if (correspondingX < 0 || correspondingX > width - 1) {
				dst.at<uchar>(y, x) = 0;
			}
			else {
				dst.at<uchar>(y, x) = src.at<uchar>(y, correspondingX);
			}
		}
	}
	imshow("strech", dst);
}

int main()
{
	system("cls");
	destroyAllWindows();
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = openImage(fname);
		Mat contur = canny(src);

		Mat src_uchar = showIntMat("Input image", src);
		Mat canny_uchar = showIntMat("Canny", contur);

		normalHough(canny_uchar, src_uchar);
		waitKey();
	}
	return 0;
}