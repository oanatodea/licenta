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
int ignoreMargin;

Mat colorSrc;
enum ArrowType {RIGHT, LEFT, UP, DOWN};

bool isInRange(int i, int j);
int dirCuantification(double value);
Mat stretch(Mat src, Point pLeft, Point pRight, Point pIntersection);
void classification(vector<Point> corners, Mat img);
void cornersClasification(vector<pair<Point, int>> cornersWithLabels, Mat img);
Mat convolution(Mat src, std::vector< std::vector<double>> filter, std::vector<double> offsetI, std::vector<double> offsetJ);
Mat closing(Mat src, const int convolutionSize);

Mat openImage(char* fname) {
	//char fname[MAX_PATH] = "C:\\Users\\Oana\\Desktop\\licenta\\srcgray.png";
	//while (openFileDlg(fname))
	//{
	//openFileDlg(fname);
	Mat graySrc;
	colorSrc = imread(fname);

	if (colorSrc.channels() == 3)
		cvtColor(colorSrc, graySrc, CV_BGR2GRAY);
	else
		colorSrc.copyTo(graySrc);

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
	ignoreMargin = max(ignoreMargin, filter.size() / 2);
	for (int i = ignoreMargin; i < height - ignoreMargin; i++) {
		for (int j = ignoreMargin; j < width - ignoreMargin; j++) {
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
			dst.at<int>(i, j) = newValue;
		}
	}
	return dst;
}

Mat gaussFiltration(Mat src) {
	std::vector< std::vector<double>> filter = { { 1.0 / 16, 1.0 / 8, 1.0 / 16 }, { 1.0 / 8, 1.0 / 4, 1.0 / 8 }, { 1.0 / 16, 1.0 / 8, 1.0 / 16 } };
	/*vector< std::vector<double>> filter = { 
	{ 1.0 / 273, 4.0 / 273, 7.0 / 273, 4.0 / 273, 1.0 / 273},
	{ 4.0 / 273, 16.0 / 273, 26.0 / 273, 16.0 / 273, 4.0 / 273 },
	{ 7.0 / 273, 26.0 / 273, 41.0 / 273, 26.0 / 273, 7.0 / 273 },
	{ 4.0 / 273, 16.0 / 273, 26.0 / 273, 16.0 / 273, 4.0 / 273 },
	{ 1.0 / 273, 4.0 / 273, 7.0 / 273, 4.0 / 273, 1.0 / 273 } };*/
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

	for (int i = ignoreMargin; i < height - ignoreMargin; i++) {
		for (int j = ignoreMargin; j < width - ignoreMargin; j++) {
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
	for (int i = ignoreMargin; i < height - ignoreMargin; i++) {
		for (int j = ignoreMargin; j < width - ignoreMargin; j++) {
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
	for (int i = ignoreMargin; i < height - ignoreMargin; i++) {
		for (int j = ignoreMargin; j < width - ignoreMargin; j++) {
			int normalizedValue = module.at<int>(i, j) / (4 * sqrt(2));
			hist[normalizedValue]++;
		}
	}
	int nrNonMuchii = (1 - p) * ((width - 2 * ignoreMargin) * (height - 2 * ignoreMargin) - hist[0]);
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

vector<Point> findPointsForCoord(Vec2f coord) {
	Point p1, p2;
	double rho = coord[0], theta = coord[1];
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;
	p1.x = cvRound(x0 + 1000 * (-b));
	p1.y = cvRound(y0 + 1000 * (a));
	p2.x = cvRound(x0 - 1000 * (-b));
	p2.y = cvRound(y0 - 1000 * (a));
	return vector < Point > {p1, p2};
}

vector<Point> findLane(Vec2f leftCoord, Vec2f rightCoord) {
	vector<Point> pointsLeft, pointsRight;
	double diag = sqrt(width * width + height * height);
	if (leftCoord[0] == NULL && rightCoord[0] == NULL) {
		pointsLeft = vector < Point > {Point(0, height), Point(width / 2 - 50, height - 200)};
		pointsRight = vector < Point > {Point(width / 2 + 30, height - 200), Point(width, height)};
	}
	else {
		if (leftCoord[0] == NULL) {
			// only left line not found
			leftCoord[1] = M_PI - rightCoord[1];
			leftCoord[0] = 0;
		}
		else {
			if (rightCoord[0] == NULL) {
				// only right line not found
				rightCoord[1] = M_PI - leftCoord[1];
				rightCoord[0] = 0;
			}
		}
		pointsLeft = findPointsForCoord(leftCoord);
		pointsRight = findPointsForCoord(rightCoord);
	}
	Point intersectionPoint = findIntersectionPoint(pointsLeft[0], pointsLeft[1], pointsRight[0], pointsRight[1]);
	return vector < Point > {pointsLeft[0], pointsRight[1], intersectionPoint};
}

vector<Point2f> findROI(vector<Vec2f> lanesCoord) {
	vector<Point> lanePoints = findLane(lanesCoord[0], lanesCoord[1]);
	Point pLeft = lanePoints[0], pRight = lanePoints[1], pIntersection = lanePoints[2];
	double yDif;
	if (pLeft.y > height) {
		yDif = abs(height - pIntersection.y);
	}
	else {
		yDif = abs(pLeft.y - pIntersection.y);
	}
	Point p1New, p2New;
	p1New.y = pIntersection.y + yDif / 5;
	p2New.y = pIntersection.y + yDif / 5;
	p1New.x = 0;
	p2New.x = width - 1;
	Point pIntersectionLeft = findIntersectionPoint(pLeft, pIntersection, p1New, p2New);
	Point pIntersectionRight = findIntersectionPoint(pRight, pIntersection, p1New, p2New);
	return vector<Point2f> { pLeft, pIntersectionLeft, pIntersectionRight, pRight };
}


vector<Vec2f> hough(Mat mat) {
	vector<Vec2f> lines;
	//empiric
	HoughLines(mat, lines, 1, M_PI / 180, 100, 0, 0);
	Vec2f leftLineExt, rightLineExt;
	leftLineExt = NULL;
	rightLineExt = NULL;
	vector<double> t;
	
	for (int i = 0; i < lines.size(); i++)
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
				if (leftLineExt[0] == NULL || rho < leftLineExt[0] || (rho == leftLineExt[0] && theta < leftLineExt[1])) {//theta < leftLineExt[1] || (theta == leftLineExt[1] && rho < leftLineExt[0])) {
					leftLineExt = lines[i];
				}
			}
			if (isInRightInterval(theta)) {
				if (rightLineExt[0] == NULL || theta > rightLineExt[1] || (theta == rightLineExt[1] && rho < rightLineExt[0])) {
					rightLineExt = lines[i];
				}
			}
		}
	}
	imshow("hough", mat);
	return vector < Vec2f > {leftLineExt, rightLineExt};
}


Point2d applyHomography(Point2d point, Mat H)
{
	Point2d ret = Point2d(-1, -1);

	const double u = H.at<double>(0, 0) * point.x + H.at<double>(0, 1) * point.y + H.at<double>(0, 2);
	const double v = H.at<double>(1, 0) * point.x + H.at<double>(1, 1) * point.y + H.at<double>(1, 2);
	const double s = H.at<double>(2, 0) * point.x + H.at<double>(2, 1) * point.y + H.at<double>(2, 2);
	if (s != 0)
	{
		ret.x = (u / s);
		ret.y = (v / s);
	}
	return ret;
}

void createMaps(Mat H, Mat H_inv, Mat& mapX, Mat& mapY, Mat& invMapX, Mat& invMapY) {
	mapX.create(height, width, CV_32F);
	mapY.create(height, width, CV_32F);
	for (int j = 0; j < height; ++j)
	{
		float* ptRowX = mapX.ptr<float>(j);
		float* ptRowY = mapY.ptr<float>(j);
		for (int i = 0; i < width; ++i)
		{
			Point2f pt = applyHomography(Point2f(i, j), H_inv);
			ptRowX[i] = pt.x;
			ptRowY[i] = pt.y;
		}
	}

	invMapX.create(height, width, CV_32F);
	invMapY.create(height, width, CV_32F);

	for (int j = 0; j < height; ++j)
	{
		float* ptRowX = invMapX.ptr<float>(j);
		float* ptRowY = invMapY.ptr<float>(j);
		for (int i = 0; i < width; ++i)
		{
			Point2f pt = applyHomography(Point2f(i, j), H);
			ptRowX[i] = pt.x;
			ptRowY[i] = pt.y;
		}
	}
}


void applyHomography(const Mat inputImg, Mat& dstImg, Mat mapX, Mat mapY)
{
	// Generate IPM image from src
	remap(inputImg, dstImg, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT);
}

Mat ipm(Mat src, vector<Point2f> pointsROI) {
	/*double yDif;
	if (pLeft.y > height) {
		yDif = abs(height - pIntersection.y);
	}
	else {
		yDif = abs(pLeft.y - pIntersection.y);
	}
	Point p1New, p2New;
	p1New.y = pIntersection.y + yDif / 5;
	p2New.y = pIntersection.y + yDif / 5;
	p1New.x = 0;
	p2New.x = width - 1;
	Point pIntersectionLeft = findIntersectionPoint(pLeft, pIntersection, p1New, p2New);
	Point pIntersectionRight = findIntersectionPoint(pRight, pIntersection, p1New, p2New);
	vector<Point2f> pointsToStretch{ pLeft, pIntersectionLeft, pIntersectionRight, pRight };
	*/
	vector<Point2f> dstPoints;
	dstPoints.push_back(Point2f(0, height));
	dstPoints.push_back(Point2f(0, 0));
	dstPoints.push_back(Point2f(width, 0));
	dstPoints.push_back(Point2f(width, height));

	Mat m_H, m_H_inv;
	assert(pointsROI.size() == 4 && dstPoints.size() == 4);
	m_H = getPerspectiveTransform(pointsROI, dstPoints);
	m_H_inv = m_H.inv();

	Mat mapX, mapY, dst;
	Mat invMapX, invMapY;
	createMaps(m_H, m_H_inv, mapX, mapY, invMapX, invMapY);

	applyHomography(src, dst, mapX, mapY);

	return dst;
}

Mat binarization(Mat src) {
	Mat dst = src.clone();
	int threshold = 170;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width;  j++) {
			if (src.at<uchar>(i, j) > threshold) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}


Mat dilation(Mat src, vector<int> convMatrixI, vector<int> convMatrixJ) {
	Mat dst = src.clone();
	int i, j, convIndexI, convIndexJ;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 255) {
				for (convIndexI = 0; convIndexI < convMatrixI.size(); convIndexI++) {
					for (convIndexJ = 0; convIndexJ < convMatrixJ.size(); convIndexJ++) {
						int newI = i + convMatrixI[convIndexI];
						int newJ = j + convMatrixJ[convIndexJ];
						if (isInRange(newI, newJ)) {
							dst.at<uchar>(newI, newJ) = 255;
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat erosion(Mat src, vector<int> convMatrixI, vector<int> convMatrixJ) {
	Mat dst = src.clone();
	int i, j, convIndexI, convIndexJ;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 255) {
				bool hasBlackNeighbours = false;
				for (convIndexI = 0; convIndexI < convMatrixI.size() && !hasBlackNeighbours; convIndexI++) {
					for (convIndexJ = 0; convIndexJ < convMatrixJ.size() && !hasBlackNeighbours; convIndexJ++) {
						int newI = i + convMatrixI[convIndexI];
						int newJ = j + convMatrixJ[convIndexJ];
						if (isInRange(newI, newJ)) {
							if (src.at<uchar>(newI, newJ) == 0) {
								hasBlackNeighbours = true;
							}
						}
					}
				}
				if (hasBlackNeighbours) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return dst;
}

Mat opening(Mat src, const int convolutionSize) {
	vector<int> convMatrixI;
	vector<int> convMatrixJ;
	for (int i = -convolutionSize / 2; i <= convolutionSize / 2; i++) {
		convMatrixI.push_back(i);
		convMatrixJ.push_back(i);
	}
	src = erosion(src, convMatrixI, convMatrixJ);
	src = dilation(src, convMatrixI, convMatrixJ);
	//imshow("open", src);
	return src;
}

Mat closing(Mat src, const int convolutionSize) {
	vector<int> convMatrixI;
	vector<int> convMatrixJ;
	for (int i = -convolutionSize / 2; i <= convolutionSize / 2; i++) {
		convMatrixI.push_back(i);
		convMatrixJ.push_back(i);
	}
	src = dilation(src, convMatrixI, convMatrixJ);
	src = erosion(src, convMatrixI, convMatrixJ);
	//imshow("close", src);
	return src;
}

Mat etichetare(Mat src, Mat opened) {
	Mat dst = src.clone();
	int eticheta = 255;
	vector<int> offsetI = { -1, 0, 1 };
	vector<int> offsetJ = { -1, 0, 1 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = 0;
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 255 && opened.at<uchar>(i,j) == 255 && dst.at<uchar>(i,j) == 0) {
				eticheta--;
				deque<Point> points;
				Point p;
				p.x = j;
				p.y = i;
				points.push_back(p);
				while (!points.empty()) {
					Point p = points.front();
					points.pop_front();
					dst.at<uchar>(p.y, p.x) = eticheta;
					for (int indexI = 0; indexI < offsetI.size(); indexI++) {
						for (int indexJ = 0; indexJ < offsetJ.size(); indexJ++) {
							// todo where is vector[] change with at()
							// todo point*
							int newI = p.y + offsetI.at(indexI);
							int newJ = p.x + offsetJ.at(indexJ);
							if (isInRange(newI, newJ)) {
								if (src.at<uchar>(newI, newJ) == 255 && dst.at<uchar>(newI, newJ) == 0) {
									Point p;
									p.x = newJ;
									p.y = newI;
									points.push_back(p);
									dst.at<uchar>(p.y, p.x) = eticheta;
								}
							}
						}
					}
				}
			}
		}
	}
	//imshow("etichetare", dst);
	return dst;
}

int transversalMarking(Mat src, int threshold, int minLineWidth) {
	vector<int> whitePixels;
	for (int i = 0; i < height; i++) {
		int sum = 0;
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) != 0) {
				sum++;
			}
		}
		whitePixels.push_back(sum);
	}
	int noOfLinesFound = 0;
	int noOfWhitePixels = 0;
	int whitePixelsError = 10;
	for (int i = 0; i < whitePixels.size(); i++) {
		if (whitePixels.at(i) >= threshold) {
			noOfLinesFound++;
			if (noOfLinesFound == 0) {
				noOfWhitePixels = whitePixels.at(i);
			}
			if (noOfLinesFound >= minLineWidth) {
				return i - noOfLinesFound + 1;
			}
		}
		else {
			if (noOfLinesFound != 0) {
				noOfLinesFound = 0;
				noOfWhitePixels = 0;
			}
		}
	}
	return -1;
}

Mat otherMarkingsElimination(Mat src, int threshold, vector<Point>& markingPoints, int minLineWidth) {
	Mat dst = src.clone();
	vector<int> offsetI = { -1, 0, 1 };
	vector<int> offsetJ = { -1, 0, 1 };
	int firstLineMarking = transversalMarking(src, threshold, minLineWidth);
	int minY = height, maxY = 0, minX = width, maxX = 0;
	if (firstLineMarking != -1 && firstLineMarking < height / 2) {
		int i = firstLineMarking;
		while (i < height && i <= firstLineMarking + 3) {
			for (int j = 0; j < width; j++) {
				if (dst.at<uchar>(i, j) != 0) {
					minY = height, maxY = 0, minX = width, maxX = 0;
					deque<Point> points;
					Point p;
					p.x = j;
					p.y = i;
					points.push_back(p);
					while (!points.empty()) {
						Point p = points.front();
						points.pop_front();
						dst.at<uchar>(p.y, p.x) = 0;
						if (p.x < minX) {
							minX = p.x;
						}
						if (p.x > maxX) {
							maxX = p.x;
						}
						if (p.y < minY) {
							minY = p.y;
						}
						if (p.y > maxY) {
							maxY = p.y;
						}
						for (int indexI = 0; indexI < offsetI.size(); indexI++) {
							for (int indexJ = 0; indexJ < offsetJ.size(); indexJ++) {
								int newI = p.y + offsetI.at(indexI);
								int newJ = p.x + offsetJ.at(indexJ);
								if (isInRange(newI, newJ)) {
									if (dst.at<uchar>(newI, newJ) != 0) {
										Point p;
										p.x = newJ;
										p.y = newI;
										points.push_back(p);
										dst.at<uchar>(p.y, p.x) = 0;
									}
								}
							}
						}
					}
				}
			}
			i++;
		}
		markingPoints.push_back(Point(minX, maxY));
		markingPoints.push_back(Point(minX, minY));
		markingPoints.push_back(Point(maxX, minY));
		markingPoints.push_back(Point(maxX, maxY));
	}
	return dst;
}

Mat harris(Mat src, Mat grey_image) {
	Mat dst;
	dst = src.clone();
	vector<Point> corners;
	vector<pair<Point, int>> cornersWithLabels;
	goodFeaturesToTrack(dst, corners, 20, 0.1, 3, noArray(), 5, true, 0.04);
	while(!corners.empty()) {
		Point p = corners.back();
		corners.pop_back();
		int label = src.at<uchar>(p.y, p.x);
		if (label == 0) {
			bool labelFound = false;
			// find label through neighbours
			for (int i = -2; i <= 2 && !labelFound; i++) {
				for (int j = -2; j <= 2 && !labelFound; j++) {
					int newI = p.y + i;
					int newJ = p.x + j;
					if (isInRange(newI, newJ)) {
						if (src.at<uchar>(newI, newJ) != 0) {
							label = src.at<uchar>(newI, newJ);
							labelFound = true;
						}
					}
				}
			}
		}
		cornersWithLabels.push_back(make_pair(p, label ));
		
		if (label != 0) {
			circle(grey_image, p, 5, Scalar(0), 2, 8, 0);
		}
	}
	Mat color_img;
	cvtColor(grey_image, color_img, CV_GRAY2BGR);
	cornersClasification(cornersWithLabels, color_img);
	imshow("harris", grey_image);
	//imshow("color", color_img);
	return color_img;
}

void cornersClasification (vector<pair<Point, int>> cornersWithLabels, Mat img) {
	for (int i = 0; i < cornersWithLabels.size(); i++) {
		if (cornersWithLabels.at(i).second != -1) {
			vector<Point> cornersForLabel;
			cornersForLabel.push_back(cornersWithLabels.at(i).first);
			int currentLabel = cornersWithLabels.at(i).second;
			for (int j = i + 1; j < cornersWithLabels.size(); j++) {
				if (cornersWithLabels.at(j).second == currentLabel){
					cornersForLabel.push_back(cornersWithLabels.at(j).first);
					cornersWithLabels.at(j).second = -1;
				}
			}
			cornersWithLabels.at(i).second = -1;
			if (cornersForLabel.size() >= 7) {
				classification(cornersForLabel, img);
			}
		}
	}
}

void drawRectangle(vector<Point> corners, Mat color_img, ArrowType arrowType) {
	int minX = width, minY = height, maxX = 0, maxY = 0;
	for (int i = 0; i < corners.size(); i++) {
		Point p = corners.at(i);
		if (p.x < minX) {
			minX = p.x;
		}
		if (p.x > maxX) {
			maxX = p.x;
		}
		if (p.y < minY) {
			minY = p.y;
		}
		if (p.y > maxY) {
			maxY = p.y;
		}
	}
	vector<Point> points;
	points.push_back(Point(minX - 1, maxY + 1));
	points.push_back(Point(minX + 1, minY - 1));
	points.push_back(Point(maxX + 1, minY - 1));
	points.push_back(Point(maxX + 1, maxY + 1));
	Scalar color = { 255, 255, 255 };
	if (arrowType == LEFT) {
		color = { 255, 0, 0 };
	}
	if (arrowType == RIGHT) {
		color = { 0, 0, 255 };
	}
	if (arrowType == UP) {
		color = { 0, 255, 0 };
	}
	polylines(color_img, points, true, color, 2, CV_AA);
}

bool isLeft(Point a, Point b, Point c){
	if (c.x < a.x || c.x < b.x) {
		return true;
	}
	return false;
}

bool isUp(Point a, Point b, Point c) {
	if (c.y < a.y || c.y < b.y) {
		return true;
	}
	return false;
}

bool contains(vector<int> vector, int x) {
	for (int i = 0; i < vector.size(); i++) {
		if (vector.at(i) == x) {
			return true;
		}
	}
	return false;
}

ArrowType findOrientation(vector<Point> corners, vector<int> indexCollinearPoints) {
	Point p1 = corners.at(indexCollinearPoints.at(0));
	Point p2 = corners.at(indexCollinearPoints.at(1));
	const double errorAngle = 30;
	double angle = abs(atan2(abs(p2.y - p1.y), abs(p2.x - p1.x)) * 180 / M_PI);
	if ((180 - angle) <= errorAngle || angle <= errorAngle) {
		//horizontal line
		int pointsInUpperPlane = 0, pointsInLowerPlane = 0;
		for (int i = 0; i < corners.size(); i++) {
			if (!contains(indexCollinearPoints, i)) {
				if (isUp(p1, p2, corners.at(i))) {
					pointsInUpperPlane++;
				}
				else {
					pointsInLowerPlane++;
				}
			}
		}
		if (pointsInUpperPlane == 1) {
			return UP;
		}
		if (pointsInLowerPlane == 1) {
			return DOWN;
		}
	}
	if (abs(angle - 90) <= errorAngle) {
		//vertical line
		int pointsInLeftPlane = 0, pointsInRightPlane = 0;
		for (int i = 0; i < corners.size(); i++) {
			if (!contains(indexCollinearPoints, i)) {
				if (isLeft(p1, p2, corners.at(i))) {
					pointsInLeftPlane++;
				}
				else {
					pointsInRightPlane++;
				}
			}
		}
		if (pointsInLeftPlane == 1) {
			return LEFT;
		}
		if (pointsInRightPlane == 1) {
			return RIGHT;
		}
	}
}

void classification(vector<Point> corners, Mat color_img) {
	const double distThreshold = 7;
	const double distBetweenPointsThreshold = 20;
	const int inlinersThreshold = 4;
	int currentInliners;
	bool inlinersFound = false;
	for (int i = 0; i < corners.size() && !inlinersFound; i++) {
		double a, b, c;
		Point p1 = corners.at(i);
		Point p2;
		p2 = corners.at((i + 1)%corners.size());
		currentInliners = 2;
		double distBetweenPoints = sqrt(pow(p1.x - p2.x, 2.0) + pow(p1.y - p2.y, 2.0));
		if (distBetweenPoints <= distBetweenPointsThreshold) {
			vector<int> indexInliners;
			indexInliners.push_back(i);
			indexInliners.push_back((i + 1) % corners.size());
			a = p1.y - p2.y;
			b = p2.x - p1.x;
			c = p1.x * p2.y - p2.x * p1.y;
			double num = sqrt(a * a + b * b);
			for (int j = 0; j < corners.size() && !inlinersFound; j++) {
				if (j != i && j != ((i + 1) % corners.size())) {
					Point p = corners.at(j);
					double dist = abs(a * p.x + b * p.y + c) / num;
					if (dist <= distThreshold) {
						currentInliners++;
						indexInliners.push_back(j);
						if (currentInliners == inlinersThreshold) {
							inlinersFound = true;
							ArrowType arrowType = findOrientation(corners, indexInliners);
							drawRectangle(corners, color_img, arrowType);
						}
					}
				}
			}
		}
	}
}

bool checkMarking(vector<Point> markingPoints, int maxWidth) {
	if (!markingPoints.empty() && abs(markingPoints[0].y - markingPoints[1].y) > maxWidth) {
		return false;
	}
	return true;
}

void findLineOnROI(Mat src, vector<Point2f> pointsROI) {
	Mat dst = binarization(src);
	dst = closing(dst, 3);
	//dst = opening(dst, 3);
	Mat color_img;
	cvtColor(src, color_img, CV_GRAY2BGR);
	imshow("bin", dst);
	//left line
	Point p1 = pointsROI[0];
	Point p2 = pointsROI[1];
	int minY = min(p1.y, p2.y);
	int maxY = max(p1.y, p2.y);
	for (int y = minY; y < maxY; y++) {
		double x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
		if (isInRange(y, x)) {
			for (int xOffset = x - 7; xOffset <= x + 7; xOffset++) {
				if (isInRange(y, xOffset)) {
					if (dst.at<uchar>(y, xOffset) == 255) {
						Vec3b color = color_img.at<Vec3b>(Point(xOffset, y));
						color[0] = 0;
						color[1] = 255;
						color[2] = 0;
						color_img.at<Vec3b>(Point(xOffset, y)) = color;
					}
				}
			}
		}
	}

	//right line
	p1 = pointsROI[2];
	p2 = pointsROI[3];
	minY = min(p1.y, p2.y);
	maxY = max(p1.y, p2.y);
	for (int y = minY; y < maxY; y++) {
		double x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
		if (isInRange(y, x)) {
			for (int xOffset = x - 7; xOffset <= x + 7; xOffset++) {
				if (isInRange(y, xOffset)) {
					if (dst.at<uchar>(y, xOffset) == 255) {
						Vec3b color = color_img.at<Vec3b>(Point(xOffset, y));
						color[0] = 0;
						color[1] = 255;
						color[2] = 0;
						color_img.at<Vec3b>(Point(xOffset, y)) = color;
					}
				}
			}
		}
	}

	imshow("lanes", color_img);

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


		vector<Vec2f> lanesCoord = hough(canny_uchar);
		vector<Point2f> pointsROI = findROI(lanesCoord);

		Mat dst = src_uchar.clone();
		vector<Point> auxPoints{ pointsROI[0], pointsROI[1], pointsROI[2], pointsROI[3] };
		polylines(dst, auxPoints, true, Scalar(255, 255, 255), 2, CV_AA);
		imshow("ROI", dst);

		findLineOnROI(src_uchar, pointsROI);

		Mat stretched = ipm(src_uchar, pointsROI);
		Mat binarImage = binarization(stretched);
		
		const int thresholdForZebra = width / 3;
		const int thresholdForLine =  width / 2;
		vector<Point> zebraPoints, stopLinePoints;
		Mat closed = closing(binarImage, 3);
	//	imshow("binar", closed);
		vector<vector<Point>> laneMarkingPoints;
		//Mat withoutLaneMarkings = laneMarkingDetection(closed, laneMarkingPoints);
		Mat withoutMarkings = otherMarkingsElimination(closed, thresholdForLine, stopLinePoints, 5);
		withoutMarkings = otherMarkingsElimination(withoutMarkings, thresholdForZebra, zebraPoints, 7);
		Mat opened = opening(withoutMarkings, 5);
		Mat etichetata = etichetare(withoutMarkings, opened);
		Mat color_img = harris(etichetata, stretched);
		Mat laneImg = color_img.clone();
		if (checkMarking(stopLinePoints, 30)) {
			polylines(color_img, stopLinePoints, true, Scalar(0, 255, 255), 1, 8, 0);
		}
		if (checkMarking(zebraPoints, height/2)) {
			polylines(color_img, zebraPoints, true, Scalar(255, 255, 0), 1, 8, 0);
		}
		//while (!laneMarkingPoints.empty()) {
		//	polylines(laneImg, laneMarkingPoints.back(), true, Scalar(0, 255, 255), 1, 8, 0);
	//		laneMarkingPoints.pop_back();
	//	}
	//	imshow("detectie", color_img);
		//imshow("lane", laneImg);
		waitKey();
	}
	return 0;
}