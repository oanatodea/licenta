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
Mat borderTracing(Mat src);

Mat openImage() {
	char fname[MAX_PATH] = "C:\\Users\\Oana\\Desktop\\licenta\\srcgray.png";
	//while (openFileDlg(fname))
	//{
	openFileDlg(fname);
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
			if (dst.at<int>(i, j) != 0) {
				int newI1 = i + offsetI[cuantifiedDirection];
				int newJ1 = j + offsetJ[cuantifiedDirection];
				int newI2 = i - offsetI[cuantifiedDirection];
				int newJ2 = j - offsetJ[cuantifiedDirection];
				if (isInRange(newI1, newJ1)) {
					if (dst.at<int>(i, j) <= dst.at<int>(newI1, newJ1)) {
						dst.at<int>(i, j) = 0;
					}
				}
				if (isInRange(newI2, newJ2)) {
					if (dst.at<int>(i, j) <= dst.at<int>(newI2, newJ2)) {
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

int searchForLeft(Mat src, int lastLeftJ, int i) {
	int currentJ = lastLeftJ;
	while (src.at<int>(i, currentJ) >= 255) {
		currentJ++;
	}
	return currentJ;
}

int searchForRight(Mat src, int lastRightJ, int i) {
	int currentJ = lastRightJ;
	while (src.at<int>(i, currentJ) >= 255) {
		currentJ--;
	}
	return currentJ;
}

Mat ipm(Mat src) {

	Mat dst = src.clone();

	for (int i = 0; i < height; i++)  {
		for (int j = 0; j < width; j++) {
			dst.at<int>(i, j) = 0;
		}
	}
	//start for bottom
	// start j from middle
	int middle = width / 2;
	int leftJ = middle, rightJ = middle;
	int startLine = height - 2;
	boolean found = false;
	while (startLine >= 0 && !found) {
		leftJ = middle;
		while (src.at<int>(startLine, leftJ) != 255 && leftJ > 0) {
			leftJ--;
		}
		if (leftJ != 0) {
			found = true;
		}
		else {
			startLine--;
		}
	}
	while (src.at<int>(startLine, rightJ) != 255 && rightJ < width - 1) {
		rightJ++;
	}
	int dLeft = 0;
	int dRight = 0;
	for (int i = startLine - 1; i >= height / 2; i--) {
		int newLeftJ = searchForLeft(src, leftJ, i);
		dLeft = dLeft + newLeftJ - leftJ;
		leftJ = newLeftJ;
		int newRightJ = searchForRight(src, rightJ, i);
		dRight = dRight + rightJ - newRightJ;
		rightJ = newRightJ;

		//moveLeft
		for (int j = 0; j <= leftJ - dLeft; j++) {
			dst.at<int>(i, j) = src.at<int>(i, j + dLeft);
		}
		for (int j = leftJ - dLeft + 1; j <= leftJ; j++) {
			dst.at<int>(i, j) = 0;
		}

		//moveRight
		for (int j = width - 1; j >= rightJ + dRight; j--) {
			dst.at<int>(i, j) = src.at<int>(i, j - dRight);
		}
		for (int j = rightJ + dRight - 1; j >= rightJ; j--) {
			dst.at<int>(i, j) = 0;
		}
	}
	return dst;
}

int computeI(int i, int dir) {
	switch (dir) {
	case 0: return i;
	case 1: return i - 1;
	case 2: return i - 1;
	case 3: return i - 1;
	case 4: return i;
	case 5: return i + 1;
	case 6: return i + 1;
	case 7: return i + 1;
	}
}

int computeJ(int j, int dir) {
	switch (dir) {
	case 0: return j + 1;
	case 1: return j + 1;
	case 2: return j;
	case 3: return j - 1;
	case 4: return j - 1;
	case 5: return j - 1;
	case 6: return j;
	case 7: return j + 1;
	}
}

boolean isStraightLine(vector<int> dir) {
	if (dir.size() == 0) {
		return false;
	}
	for (int i = 0; i < dir.size(); i++) {
		if (dir.at(i) > 0 && dir.at(i) < 4) {
			return false;
		}
	}
	return true;
}

Mat borderTracing(Mat src) {
	Mat dst = src.clone();
	Mat lines = src.clone();
	vector<int> indexI, indexJ, dir;
	int index;
	int found = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<int>(i, j) = 0;
			lines.at<int>(i, j) = 0;
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<int>(i, j) == 255 && dst.at<int>(i, j) == 0 && (isInRange(i, j - 1) && src.at<int>(i, j - 1) == 0)) {
				found++;
				indexI.clear();
				indexJ.clear();
				dir.clear();
				indexI.push_back(i);
				indexJ.push_back(j);
				index = 0;
				dst.at<int>(indexI.at(index), indexJ.at(index)) = 255;
				boolean forceStop = false;
				while (!forceStop && (index == 0 || index == 1 || !((indexI.at(index) == indexI.at(1)) && (indexJ.at(index) == indexJ.at(1)) && (indexI.at(index - 1) == indexI.at(0)) && (indexJ.at(index - 1) == indexJ.at(0))))) {
					int direction;
					if (index == 0) {
						direction = 5;
					} else {
						if (dir.at(index - 1) % 2 == 0) {
							direction = (dir.at(index - 1) + 7) % 8;
						}
						else {
							direction = (dir.at(index - 1) + 6) % 8;
						}
					}
					int startDir = direction;
					do {
						int newI = computeI(indexI.at(index), direction);
						int newJ = computeJ(indexJ.at(index), direction);
						if (!isInRange(newI, newJ) || src.at<int>(newI, newJ) == 0) {
							direction = (direction + 1) % 8;
						}
						else {
							break;
						}
						if (direction == startDir) {
							forceStop = true;
							break;
						}

					} while (true);

					if (!forceStop) {
						//add next point
						dir.push_back(direction);
						indexI.push_back(computeI(indexI.at(index), direction));
						indexJ.push_back(computeJ(indexJ.at(index), direction));
						dst.at<int>(indexI.at(index + 1), indexJ.at(index + 1)) = 255;
						index++;
					}
				}
				if (isStraightLine(dir)) {
					//draw line
					for (int indexes = 0; indexes < indexI.size(); indexes++) {
						lines.at<int>(indexI.at(indexes), indexJ.at(indexes)) = 255;
					}
				}
			}
		}
	}
	//imshow("lines", lines);
	return dst;
}

Mat showIntMat(String message, Mat src) {
	Mat dst = Mat(height, width, DataType<uchar>::type);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = (uchar)src.at<int>(i, j);
		}
	}
	imshow(message, dst);
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

struct peak{
	int t, r;
	double h;
	boolean peak::operator < (const peak &p2) const { return h > p2.h; };
};

boolean equals(peak p1, peak p2) {
	const int tolerance = 2;
	if ((p1.h >= (p2.h - tolerance)) && (p1.h <= (p2.h + tolerance))) {
		return true;
	}
	if ((p2.h >= (p1.h - tolerance)) && (p2.h <= (p1.h + tolerance))) {
		return true;
	}
	return false;
}

void drawLine(peak p, Mat img) {
	// drawing a line from point (x1,y1) to point (x2,y2)
	int x1 = 0;
	int y1 = (p.r - x1 * cos(p.t * M_PI / 180)) / sin(p.t * M_PI / 180);
	int x2 = width - 1;
	int y2 = (p.r - x2 * cos(p.t * M_PI / 180)) / sin(p.t * M_PI / 180);
	line(img, Point(x1, x2), Point(x2, y2), CvScalar(255, 0, 0, 0), 1, 8, 0);
}

void drawLines(double *H, Mat img)
{
	const int minVotes = 30;
	double diag = sqrt(height * height + width * width);
	for (int t = 0 ; t < 360; t++) {
		//priority_queue <peak> peaks;
		for (int r = 0; r < 10; r++) {
			if (H[r * 360 + t] > 100) {
				double a = cos(t * M_PI / 180), b = sin(t * M_PI / 180);
				double x0 = a*r, y0 = b*r;
				Point pt1, pt2;
				pt1.x = cvRound(x0 + 1000 * (-b));
				pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b));
				pt2.y = cvRound(y0 - 1000 * (a));
				line(img, pt1, pt2, Scalar(255, 255, 255), 1, CV_AA);
			}



		/*	if (H[r * 360 + t] >= minVotes) {
				peak p;
				p.t = t;
				p.r = r;
				p.h = H[r * 360 + t];
				peaks.push(p);
			}
		}
		peak current, next;
		while (peaks.size() >= 2) {
			current = peaks.top();
			peaks.pop();
			next = peaks.top();
			//if (equals(current, next)) {
				drawLine(current, img);
				//drawLine(next, img);
			//}*/
		}
	}
	imshow("hough", img);
}

void hough(Mat src) {
	double diag = sqrt(height * height + width * width);
	const int hSize = 360 * diag;
	double *H = new double[hSize];
	memset(H, 0, hSize);
	for (int i = height/2; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 255) {
				computeHough(j, i, H);
			}
		}
	}
	drawLines(H, src);
}

void anotherHough(Mat mat) {
	vector<Vec2f> lines;
	HoughLines(mat, lines, 1, M_PI / 180, 100, 0, 0);
	vector<Vec2f> letfLine, rightLine;
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		double theta_grades = theta * 180/ M_PI;
		if ((theta_grades < 60 && theta_grades > 30) || (theta_grades > 120 && theta_grades < 150)) {
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(mat, pt1, pt2, Scalar(255, 255, 255), 1, CV_AA);
		}
	}
	imshow("hough", mat);
}

int main()
{
	system("cls");
	destroyAllWindows();
	Mat src = openImage();
	Mat contur = canny(src);
	//Mat inverse = ipm(contours);
	//Mat borders = borderTracing(contur);

	showIntMat("Input image", src);
	Mat canny_uchar = showIntMat("Canny", contur);
	
	anotherHough(canny_uchar);
	waitKey();
		
	return 0;
}