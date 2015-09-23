#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <vector>
#include <dlib/optimization.h>
#include <math.h>

#define OPT 1
#define SMP 0
#define THRESH 160
#define THR1 0.5

using namespace cv;
using namespace std;
using namespace dlib;

typedef matrix<double,0,1> column_vector;
IplImage* imgSrc;
IplImage* imgTrg;
IplImage* imgWpd;

IplImage* imgSrcCp;
IplImage* imgTrgCp;
IplImage* imgWpdCp;
int counter;
int *pts;
int num_pts;

class warp
{
public:
	warp(int cps);
	~warp(void);
	std::vector<double> Eval(double u, double v);
	void warpImg(IplImage* img1, IplImage* imgRes);
	double warpedSSD(IplImage* img1, IplImage* imgRes, double x, double y, int r);
	double warpedSSDfull(IplImage* img1, IplImage* imgRes);
	void simpOpt(IplImage* imgS, IplImage* imgT, int rad);
	int simpOptSg(IplImage* imgS, IplImage* imgT, int rad);
	void drawCpt(IplImage* img1);

	int m;
	int n; 
	double *cps_x;
	double *cps_y;

	double *dsp_x;
	double *dsp_y;

	double *localSSD;
	bool *checked;
};

warp *wp;

warp::warp(int cps)
{
	/*
	string warp_func0 = "warping function9.wrf"; // WARP ZERO
	FILE *fp0 = fopen(warp_func0.c_str(), "r");
	char buffer0[100];
	fscanf(fp0, "%s", buffer0);
	fscanf(fp0, "%d%d", &m, &n); */
	
	// 11 , 20
	n=cps;
	m=cps;
	double betw = 1 / ((double) cps-2.0);

	//printf(" %d ", betw);

	cps_x = NULL;
	if( cps_x )
		free( cps_x );
	cps_x = (double *)malloc((n+1)*(m+1)*sizeof(double));
	memset(cps_x,0,sizeof(cps_x));

	cps_y = NULL;
	if( cps_y )
		free( cps_y );
	cps_y = (double *)malloc((n+1)*(m+1)*sizeof(double));
	memset(cps_y,0,sizeof(cps_y));

	dsp_x = NULL;
	if( dsp_x )
		free( dsp_x );
	dsp_x = (double *)malloc((n+1)*(m+1)*sizeof(double));
	memset(dsp_x,0,sizeof(dsp_x));

	dsp_y = NULL;
	if (dsp_y)
		free(dsp_y);
	dsp_y = (double *)malloc((n + 1)*(m + 1)*sizeof(double));
	memset(dsp_y, 0, sizeof(dsp_y));

	localSSD = NULL;
	if (localSSD)
		free(localSSD);
	localSSD = (double *)malloc((n + 1)*(m + 1)*sizeof(double));
	memset(localSSD, 0, sizeof(localSSD));

	checked = NULL;
	if (checked)
		free(checked);
	checked = (bool *)malloc((n + 1)*(m + 1)*sizeof(bool));
	memset(checked, 0, sizeof(checked));

	double inc_x = -betw;
	for (int i=0; i<=n; i++) {
		double inc_y = -betw; //-0.111;		
		for (int j=0; j<=m; j++) {
			//double x,y;
			//fscanf(fp0, "%lf%lf", &x, &y);
			//cps_x[i*(m+1) + j]=x;
			//cps_y[i*(m+1) + j]=y;
			
			cps_x[i*(m+1) + j]=inc_x;
			cps_y[i*(m+1) + j]=inc_y;

			dsp_x[i*(m+1) + j]=0.0;
			dsp_y[i*(m + 1) + j]=0.0;

			localSSD[i*(m + 1) + j] = -1.0;
			checked[i*(m + 1) + j] = false;

			inc_y+=betw; //0.111;
		}
		inc_x+=betw; //0.111;
	}

	//fclose(fp0);

	/*
	for (int i=0; i<=n; i++) {
		for (int j=0; j<=m; j++) {
			printf(" %f %f \n", cps_x[i*(m+1) + j], cps_y[i*(m+1) + j]);
			
		}
	}
	*/
		
}

warp::~warp(void)
{
	delete[] cps_x;
	delete[] cps_y;
}

void warp::warpImg(IplImage* img1, IplImage* imgRes) {
	int width = img1->width; 
	int height = img1->height;
	int step = img1->widthStep;
	//printf (" { %d %d }", width, step);

	// making the image black
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			CvScalar s;
			s.val[0]=0.0;
			s.val[1]=255.0;
			s.val[2]=0.0;
			cvSet2D(imgRes,i,j,s);
			/*
			imgRes->imageData[3*(i*width + j)]=0;
			imgRes->imageData[3*(i*width + j)+1]=(char) 255;
			imgRes->imageData[3*(i*width + j)+2]=0;*/
		}
	}

	// warping img1 onto imgRes
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			double y = i/((double) height);
			double x = j/((double) width);

			std::vector<double> pt(2); 
			pt=Eval(x, y);
			double x1=pt[0];
			double y1=pt[1];

			if (x <= 0.0) x1 = 0.0;
			if (x >= 1.0) x1 = 1.0;
			if (y <= 0.0) y1 = 0.0;
			if (y >= 1.0) y1 = 1.0;
			
			int y_pix = (int) (y1*height);
			int x_pix = (int) (x1*width);
			
			if (x1 < 0.0) x_pix = 0;
			if (x1 > 1.0) {
				//if (x<0.8)printf(" { %f %d %d } " , x, i,j);
				x_pix = width-1;
			}
			if (y1 < 0.0) y_pix = 0;
			if (y1 > 1.0) y_pix = height-1;
			
			

			
			//if (j-x_pix>100 || j-x_pix<-100) printf(" ( %f %f ) ", x, pt[0]);
			//if (j<10) printf(" %d ", x_pix);
			
			imgRes->imageData[3*(y_pix*width + x_pix)]=img1->imageData[3*(i*width + j)];
			imgRes->imageData[3*(y_pix*width + x_pix)+1]=img1->imageData[3*(i*width + j)+1];
			imgRes->imageData[3*(y_pix*width + x_pix)+2]=img1->imageData[3*(i*width + j)+2];
			//if ((int) imgRes->imageData[3*(y_pix*width + x_pix)]<0) printf(" %d ", (int) imgRes->imageData[3*(y_pix*width + x_pix)]);
		}
	}
	
	// interpolate
	
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			CvScalar curpix = cvGet2D( imgRes, i, j);
			//if (i==180 && j==305) printf(" [ %f %f %f ] ", curpix.val[0], curpix.val[1], curpix.val[2]);

			if (curpix.val[0]==0 && curpix.val[1]==255.0 && curpix.val[2]==0) {
				// if the pixel is empty - do the interpolation
				CvScalar icol;
				icol.val[0]=0.0;
				icol.val[1]=0.0;
				icol.val[2]=0.0;
	
				int cnt = 0; // number of nonzero neighbouring pixels
				int inc = 1; // sort of radius
				//printf (" %d %d ", i, j);
				while (cnt==0) {
					for (int n=i-1-inc; n<=i+1+inc; n++) {
						if (n<height && n>=0) {
							for (int m=j-1-inc; m<=j+1+inc; m++) {
								if (m<width && m>=0) {
									//
									CvScalar pixel = cvGet2D( imgRes, n, m);
									if (!(pixel.val[0]==0 && pixel.val[1]==255 && pixel.val[2]==0)) {
										icol.val[0]+=pixel.val[0];
										icol.val[1]+=pixel.val[1];
										icol.val[2]+=pixel.val[2];
										cnt++;
									}
									//
								}
							}
						}
					}
					inc++;
				}

				icol.val[0]/=cnt;
				icol.val[1]/=cnt;
				icol.val[2]/=cnt;

				cvSet2D(imgRes,i,j,icol);

				//if (red>=230 || blue>=230 || green>=230) printf(" ( %d %d ) ", i, j);
			}
	
		}
	} 
	
}

double warp::warpedSSD(IplImage* img1, IplImage* imgRes, double x, double y, int r) {
	int width = img1->width; 
	int height = img1->height;
	int step = img1->widthStep;
	//printf (" { %d %d }", width, step);

	int pix_x = (int) (x * width);
	int pix_y = (int) (y * height);

	double ssd_sum = 0.0;
	
	/*
	for (int j=pix_x-r; j<pix_x+r; j++) {
		if (j>=0 && j<width) {
			for (int i=pix_y+r; i<pix_y+r; i++) { // i<=j
				if (i>=0 && i<height) {
					CvScalar s;
					s.val[0]=0.0;
					s.val[1]=255.0;
					s.val[2]=0.0;
					cvSet2D(imgRes,i,j,s);
				}
			}
		}
	}
	*/
	double cnter = 0.0;
	// warping img1 onto imgRes
	for (int i=0; i<height; i++) {
		for (int j = 0; j < width; j++) {
			if (j<(pix_x + 4 * r) && j>(pix_x - 4 * r) && i<(pix_y + 4 * r) && i>(pix_y - 4 * r)) {
				double y0 = i / ((double)height);
				double x0 = j / ((double)width);

				std::vector<double> pt(2);
				pt = Eval(x0, y0);
				double x1 = pt[0];
				double y1 = pt[1];

				if (x0 <= 0.0) x1 = 0.0;
				if (x0 >= 1.0) x1 = 1.0;
				if (y0 <= 0.0) y1 = 0.0;
				if (y0 >= 1.0) y1 = 1.0;

				int y_pix = (int)(y1*height);
				int x_pix = (int)(x1*width);

				if (x1 < 0.0) x_pix = 0;
				if (x1 > 1.0) {
					//if (x<0.8)printf(" { %f %d %d } " , x, i,j);
					x_pix = width - 1;
				}
				if (y1 < 0.0) y_pix = 0;
				if (y1 > 1.0) y_pix = height - 1;

				//if (j-x_pix>100 || j-x_pix<-100) printf(" ( %f %f ) ", x, pt[0]);
				//if (j<10) printf(" %d ", x_pix);

				if (x_pix<(pix_x + 2 * r) && x_pix>(pix_x - 2 * r) && y_pix<(pix_y + 2 * r) && y_pix>(pix_y - 2 * r)) {
					double diff = img1->imageData[3 * (i*width + j)] - imgRes->imageData[3 * (y_pix*width + x_pix)];
					ssd_sum += diff*diff;
					cnter++;
				//printf(" %d %d %d %d \n", x_pix, pix_x, y_pix, pix_y);
				}
			}
			//if ((int) imgRes->imageData[3*(y_pix*width + x_pix)]<0) printf(" %d ", (int) imgRes->imageData[3*(y_pix*width + x_pix)]);
		}
	}
	ssd_sum /= cnter;
	
	/*
	for (int j=pix_x-r; j<pix_x+r; j++) {
		if (j>=0 && j<width) {
			for (int i=pix_y+r; i<pix_y+r; i++) { // i<=j
				if (i>=0 && i<height) {
					CvScalar pixS=cvGet2D( imgRes, i, j);
					CvScalar pixT=cvGet2D( imgTrg, i, j);

					double diff0 = (pixS.val[0] - pixT.val[0]);
					double diff1 = (pixS.val[1] - pixT.val[1]);
					double diff2 = (pixS.val[2] - pixT.val[2]);
					if (!(pixS.val[0]==0.0 && pixS.val[1]==255.0 && pixS.val[2]==0.0)) ssd_sum += ( diff0*diff0 + diff1*diff1 + diff2*diff2 ) / 9.0;
				}
			}
		}
	}*/

	return ssd_sum;
}

double warp::warpedSSDfull(IplImage* img1, IplImage* imgRes) {
	int width = img1->width;
	int height = img1->height;
	int step = img1->widthStep;
	//printf (" { %d %d }", width, step);


	double ssd_sum = 0.0;

	double cnter = 0.0;
	// warping img1 onto imgRes
	for (int i = 0; i<height; i++) {
		for (int j = 0; j < width; j++) {
	
				double y0 = i / ((double)height);
				double x0 = j / ((double)width);

				std::vector<double> pt(2);
				pt = Eval(x0, y0);
				double x1 = pt[0];
				double y1 = pt[1];

				if (x0 <= 0.0) x1 = 0.0;
				if (x0 >= 1.0) x1 = 1.0;
				if (y0 <= 0.0) y1 = 0.0;
				if (y0 >= 1.0) y1 = 1.0;

				int y_pix = (int)(y1*height);
				int x_pix = (int)(x1*width);

				if (x1 < 0.0) x_pix = 0;
				if (x1 > 1.0) {
					//if (x<0.8)printf(" { %f %d %d } " , x, i,j);
					x_pix = width - 1;
				}
				if (y1 < 0.0) y_pix = 0;
				if (y1 > 1.0) y_pix = height - 1;

				//if (j-x_pix>100 || j-x_pix<-100) printf(" ( %f %f ) ", x, pt[0]);
				//if (j<10) printf(" %d ", x_pix);

				if ( x_pix<width && x_pix>=0 && y_pix<height && y_pix>=0 ) {
					double diff = img1->imageData[3 * (i*width + j)] - imgRes->imageData[3 * (y_pix*width + x_pix)];
					ssd_sum += diff*diff;
					cnter++;
					//printf(" %d %d %d %d \n", x_pix, pix_x, y_pix, pix_y);
				}
			
		}
	}
	ssd_sum /= cnter;

	return ssd_sum;
}

void get_ucbs_basis(double t, double *basis)
{
	double tt = t * t;
	double ttt = tt * t;
	basis[0] = (1.0 - 3 * t + 3 * tt - ttt) / 6.0;
	basis[1] = (3 * ttt - 6 * tt + 4) / 6.0;
	basis[2] = (-3 * ttt + 3 * tt + 3 * t + 1) / 6.0;
	basis[3] = ttt / 6.0;
}

double ssd(IplImage* imgS, IplImage* imgT) {
	int width = imgS->width; 
	int height = imgS->height;

	double ssd_sum = 0.0;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			CvScalar pixS = cvGet2D( imgS, i, j);
			CvScalar pixT = cvGet2D( imgT, i, j);

			double diff0 = (pixS.val[0] - pixT.val[0]);
			double diff1 = (pixS.val[1] - pixT.val[1]);
			double diff2 = (pixS.val[2] - pixT.val[2]);
			ssd_sum += ( diff0*diff0 + diff1*diff1 + diff2*diff2 ) / 9.0;
		}
	}
	return ssd_sum;
}

double ssd_dir(IplImage* imgS, IplImage* imgT, double x, double y, int r, int dir) {
	int width = imgS->width; 
	int height = imgS->height;

	int pix_x = (int) (x * width);
	int pix_y = (int) (y * height);

	double ssd_sum = 0.0;
	
	for (int j=0; j<r; j++) {
		for (int i=0; i<r; i++) { // i<=j
			CvScalar pixS;
			CvScalar pixT;
			if (dir==1) { // to the right
				pixS = cvGet2D( imgS, pix_y+i, pix_x+j);
				pixT = cvGet2D( imgT, pix_y+i, pix_x+j);
			}
			else if (dir==2) { // to the left
				pixS = cvGet2D( imgS, pix_y+i, pix_x-j);
				pixT = cvGet2D( imgT, pix_y+i, pix_x-j);
			}
			else if (dir==3) { // upwards
				pixS = cvGet2D( imgS, pix_y+j, pix_x+i);
				pixT = cvGet2D( imgT, pix_y+j, pix_x+i);
			}
			else if (dir==4) { // downwards
				pixS = cvGet2D( imgS, pix_y-j, pix_x+i);
				pixT = cvGet2D( imgT, pix_y-j, pix_x+i);
			}

			double diff0 = (pixS.val[0] - pixT.val[0]);
			double diff1 = (pixS.val[1] - pixT.val[1]);
			double diff2 = (pixS.val[2] - pixT.val[2]);
			ssd_sum += ( diff0*diff0 + diff1*diff1 + diff2*diff2 ) / 9.0;

			if (i!=0) {
				if (dir==1) { // to the right
					pixS = cvGet2D( imgS, pix_y-i, pix_x+j);
					pixT = cvGet2D( imgT, pix_y-i, pix_x+j);
				}
				else if (dir==2) { // to the left
					pixS = cvGet2D( imgS, pix_y-i, pix_x-j);
					pixT = cvGet2D( imgT, pix_y-i, pix_x-j);
				}
				else if (dir==3) { // upwards
					pixS = cvGet2D( imgS, pix_y+j, pix_x-i);
					pixT = cvGet2D( imgT, pix_y+j, pix_x-i);
				}	
				else if (dir==4) { // downwards
					pixS = cvGet2D( imgS, pix_y-j, pix_x-i);
					pixT = cvGet2D( imgT, pix_y-j, pix_x-i);
				}

				diff0 = (pixS.val[0] - pixT.val[0]);
				diff1 = (pixS.val[1] - pixT.val[1]);
				diff2 = (pixS.val[2] - pixT.val[2]);
				ssd_sum += ( diff0*diff0 + diff1*diff1 + diff2*diff2 ) / 9.0;
			}
		}
	}
	return ssd_sum;
}

double ssd_nb(double x, double y, int r) {
	int width = imgWpd->width; 
	int height = imgWpd->height;

	int pix_x = (int) (x * width);
	int pix_y = (int) (y * height);

	double ssd_sum = 0.0;
	
	for (int j=pix_x-r; j<pix_x+r; j++) {
		if (j>=0 && j<width) {
			for (int i=pix_y+r; i<pix_y+r; i++) { // i<=j
				if (i>=0 && i<height) {
					CvScalar pixS=cvGet2D( imgWpd, i, j);
					CvScalar pixT=cvGet2D( imgTrg, i, j);

					double diff0 = (pixS.val[0] - pixT.val[0]);
					double diff1 = (pixS.val[1] - pixT.val[1]);
					double diff2 = (pixS.val[2] - pixT.val[2]);
					ssd_sum += ( diff0*diff0 + diff1*diff1 + diff2*diff2 ) / 9.0;
				}
			}
		}
	}
	return ssd_sum;
}

void warp::simpOpt(IplImage* imgS, IplImage* imgT, int rad) {
	int width = imgS->width; 
	int height = imgS->height;
	
	double nb_x = rad/((double) width);
	double nb_y = rad/((double) height);

	for (int i=0; i<=n; i++) {
		for (int j=0; j<=m; j++) {
			double x = cps_x[i*(m+1) + j];
			double y = cps_y[i*(m+1) + j];

			if ((x+nb_x)<1.0 && (x-nb_x)>=0 && (y+nb_y)<1.0 && (y-nb_y)>=0) {

				double delta1_x = ssd_dir(imgS, imgT, x, y, rad, 1);
				double delta2_x = ssd_dir(imgS, imgT, x, y, rad, 2);
				double delta1_y = ssd_dir(imgS, imgT, x, y, rad, 3);
				double delta2_y = ssd_dir(imgS, imgT, x, y, rad, 4);

				double max = ( 255.0 * 255.0 * 4.0 * rad * rad ) / 6.0;

				double delta_x = 0; 
				if ((delta1_x + delta2_x)!=0 && SMP==1) delta_x = (delta1_x - delta2_x)/(delta1_x + delta2_x);
				else delta_x = (delta1_x - delta2_x)/max;
				
				double delta_y = 0; 
				if ((delta1_y + delta2_y)!=0 && SMP==1) delta_y = (delta1_y - delta2_y)/(delta1_y + delta2_y);
				else delta_y = (delta1_y - delta2_y)/max;

				dsp_x[i*(m+1) + j] = nb_x*delta_x;
				//printf("\n%d %d x SSD: %f %f \n ", i, j, nb_x, delta_x);
				dsp_y[i*(m+1) + j] = nb_y*delta_y;	
				//printf("\n%d %d y SSD: %f %f \n", i, j, nb_y, delta_y);

				if ((i*(m+1) + j)==50 || (i*(m+1) + j)==100 ) {
					printf(" %f %f %f %f \n",  delta1_x, delta2_x, delta1_y, delta2_y);
				}
				
				//i=n;
				//j=m;
			}
			// endif

		}
	}

	// smoothness filter
	for (int i=0; i<=n; i++) {
		for (int j=0; j<=m; j++) {
			
			double avg_x = 0.0;
			double avg_y = 0.0;
			
			int cnt = 0;
			int nb = 1;
			// average value
			for (int p=i-nb; p<=i+nb; p++) {
				for (int q=j-nb; q<=j+nb; q++) {
					if (!(p==0 && q==0) && (p>0) && (q>0) && (p<n) && (q<m)) {
						avg_x += dsp_x[p*(m+1) + q];
						avg_y += dsp_y[p*(m+1) + q];
						cnt++;
					}
				}
			}

			if (cnt!=0) {
				avg_x/= ((double) cnt);
				avg_y/= ((double) cnt);
			}

			if (abs(dsp_x[i*(m+1) + j]) > 0.005 &&  SMP==0) cps_x[i*(m+1) + j] += (2*dsp_x[i*(m+1) + j]);
			else cps_x[i*(m+1) + j] += (dsp_x[i*(m+1) + j] + avg_x)/2.0;
			if (abs(dsp_y[i*(m+1) + j]) > 0.005 &&  SMP==0) cps_y[i*(m+1) + j] += (2*dsp_y[i*(m+1) + j]);
			else cps_y[i*(m+1) + j] += (dsp_y[i*(m+1) + j] + avg_y)/2.0;
			

			//cps_x[i*(m+1) + j] += dsp_x[i*(m+1) + j];
			//cps_y[i*(m+1) + j] += dsp_y[i*(m+1) + j];
		}
	}
}

int warp::simpOptSg(IplImage* imgS, IplImage* imgT, int rad) {
	int width = imgS->width; 
	int height = imgS->height;
	
	double nb_x = rad/((double) width);
	double nb_y = rad/((double) height);

	double max_dsp = 0;
	int max_cp;

	for (int i=0; i<=n; i++) {
		for (int j=0; j<=m; j++) {
			double x = cps_x[i*(m+1) + j];
			double y = cps_y[i*(m+1) + j];

			if ((x + nb_x)<1.0 && (x - nb_x) >= 0 && (y + nb_y)<1.0 && (y - nb_y) >= 0 && (x > 0.01) && (x < 0.99) && (y > 0.01) && (y < 0.99)) {

				double delta1_x = ssd_dir(imgS, imgT, x, y, rad, 1);
				double delta2_x = ssd_dir(imgS, imgT, x, y, rad, 2);
				double delta1_y = ssd_dir(imgS, imgT, x, y, rad, 3);
				double delta2_y = ssd_dir(imgS, imgT, x, y, rad, 4);

				double max = ( 255.0 * 255.0 * 4.0 * rad * rad ) / 6.0;

				double delta_x = 0; 
				if ((delta1_x + delta2_x)!=0 && SMP==1) delta_x = (delta1_x - delta2_x)/(delta1_x + delta2_x);
				else delta_x = (delta1_x - delta2_x)/max;
				
				double delta_y = 0; 
				if ((delta1_y + delta2_y)!=0 && SMP==1) delta_y = (delta1_y - delta2_y)/(delta1_y + delta2_y);
				else delta_y = (delta1_y - delta2_y)/max;

				dsp_x[i*(m+1) + j] = nb_x*delta_x;
				//printf("\n%d %d x SSD: %f %f \n ", i, j, nb_x, delta_x);
				dsp_y[i*(m+1) + j] = nb_y*delta_y;	
				//printf("\n%d %d y SSD: %f %f \n", i, j, nb_y, delta_y);
				if ((dsp_x[i*(m + 1) + j] + dsp_y[i*(m + 1) + j]) > max_dsp && !checked[i*(m + 1) + j] ) {
					max_dsp = (dsp_x[i*(m+1) + j] + dsp_y[i*(m+1) + j]); 
					max_cp = i*(m + 1) + j;

				}

				if ((i*(m+1) + j)==50 || (i*(m+1) + j)==100 ) {
					//printf(" %f %f %f %f \n",  delta1_x, delta2_x, delta1_y, delta2_y);
				}
				
				//i=n;
				//j=m;
			}
			// endif

		}
	}

	for (int i=0; i< (m+1)*(n+1); i++) {
		if ( (dsp_x[i] + dsp_y[i]) > THR1 * max_dsp) {
			printf(" !! %d !! ", i);
			cps_x[i] += dsp_x[i];
			cps_y[i] += dsp_y[i];
		}
	}

	/*
	// smoothness filter
	for (int i=0; i<=n; i++) {
		for (int j=0; j<=m; j++) {
			
			double avg_x = 0.0;
			double avg_y = 0.0;
			
			int cnt = 0;
			int nb = 1;
			// average value
			for (int p=i-nb; p<=i+nb; p++) {
				for (int q=j-nb; q<=j+nb; q++) {
					if (!(p==0 && q==0) && (p>0) && (q>0) && (p<n) && (q<m)) {
						avg_x += dsp_x[p*(m+1) + q];
						avg_y += dsp_y[p*(m+1) + q];
						cnt++;
					}
				}
			}

			if (cnt!=0) {
				avg_x/= ((double) cnt);
				avg_y/= ((double) cnt);
			}

			if (abs(dsp_x[i*(m+1) + j]) > 0.005 &&  SMP==0) cps_x[i*(m+1) + j] += (2*dsp_x[i*(m+1) + j]);
			else cps_x[i*(m+1) + j] += (dsp_x[i*(m+1) + j] + avg_x)/2.0;
			if (abs(dsp_y[i*(m+1) + j]) > 0.005 &&  SMP==0) cps_y[i*(m+1) + j] += (2*dsp_y[i*(m+1) + j]);
			else cps_y[i*(m+1) + j] += (dsp_y[i*(m+1) + j] + avg_y)/2.0;
			

			//cps_x[i*(m+1) + j] += dsp_x[i*(m+1) + j];
			//cps_y[i*(m+1) + j] += dsp_y[i*(m+1) + j];
		}
	}
	*/
	return max_cp;
}

void warp::drawCpt(IplImage* img1){
	int width = img1->width; 
	int height = img1->height;

	for (int i=0; i<=n; i++) {
		for (int j=0; j<=m; j++) {
			int x = (int) (width * cps_x[i*(m+1) + j]);
			int y = (int)(height * cps_y[i*(m + 1) + j]);
			CvScalar pix;

			pix.val[0]=0;
			pix.val[1]=0;
			pix.val[2]=255;

			bool isIn = false;
			for (int k = 0; k < num_pts; k++) {
				if (pts[k] == (i*(m + 1) + j)) isIn = true;
			}
			if (isIn) {
				pix.val[1] = 255;
				pix.val[2] = 0;
			}
			
			//if (i==2 && j==5) {
			/*
			if ((i*(m + 1) + j) == 50 || (i*(m + 1) + j) == 38) {
				//printf(" { %f %f } ", cps_x[i*(m+1) + j], cps_y[i*(m+1) + j]);
				pix.val[1]=255;
			}
			*/
			
			if (x>=0 && y>=0 && x<width && y<height) cvSet2D(img1,y,x,pix);
		}
	}
	/*
	CvScalar pix;
	pix.val[0] = 0;
	pix.val[1] = 255;
	pix.val[2] = 0;
	cvSet2D(img1, 0, 0, pix);
	*/

}

std::vector<double> warp::Eval(double u, double v)
{
	/* double v;
	va_list ap;
	va_start(ap, u);
	v = va_arg(ap, double);
	va_end(ap); */
		
	// get local parameter values for u and v
	double uu = u * (m - 2);
	double vv = v * (n - 2);

	int k = (int)uu;	
	int l = (int)vv;	// P(k,l) is the left bottum control points.
	uu = uu - k;	
	vv = vv - l;	// get the local parameter of u and v.

	if (k == m - 2)	// special case for u is 1.0. 
	{
		k = m - 3;
		uu = 1.0;
	}
	if (l == n - 2)  // special case for v is 1.0.
	{
		l = n - 3;
		vv = 1.0;
	}
	
	static double basis_u[4], basis_v[4];
	get_ucbs_basis(uu, basis_u);
	get_ucbs_basis(vv, basis_v);

	std::vector<double> pt(2);
	
	pt[0]=0;
	pt[1]=0;
	
	for (int i = 0; i < 4; ++i)
	{
		int idx = (k + i) * (n + 1) + l;
		
		pt[0] += basis_u[i] * (cps_x[idx] * basis_v[0] + cps_x[idx + 1] * basis_v[1] + cps_x[idx + 2] * basis_v[2] + cps_x[idx + 3] * basis_v[3]);
		pt[1] += basis_u[i] * (cps_y[idx] * basis_v[0] + cps_y[idx + 1] * basis_v[1] + cps_y[idx + 2] * basis_v[2] + cps_y[idx + 3] * basis_v[3]);
	}
	
	return pt;
}

class test_function
{
public:
    test_function ()
    {

    }

    double operator() ( const column_vector& arg) const
    {
		counter++;
		printf("%d ", counter);
		//warp *wp = new warp(11);

		double midX = 0;
		double midY = 0;
		double midDspX = 0;
		double midDspY = 0;
		
		for (int i = 0; i<num_pts; i++) {
			wp->cps_x[pts[i]] = arg(2 * i);
			wp->cps_y[pts[i]] = arg(2 * i + 1);
			midX += arg(2 * i);
			midY += arg(2 * i + 1);
			midDspX += wp->cps_x[pts[i]] - wp->dsp_x[pts[i]];
			midDspY += wp->cps_y[pts[i]] - wp->dsp_y[pts[i]];
			printf(" %d %f %f \n", pts[i], wp->cps_x[pts[i]], wp->cps_y[pts[i]]);
		}
		midX /= (double)num_pts;
		midY /= (double)num_pts;
		midDspX /= (double)num_pts;
		midDspY /= (double)num_pts;

		//wp->warpImg(imgSrc, imgWpd);
		double diff = wp->warpedSSD(imgSrcCp, imgTrgCp, midX, midY, ((int)(0.22*imgSrc->width)));

		/*
		for (int i=0; i<num_pts; i++) {
			diff+=wp->warpedSSD(imgSrcCp, imgWpdCp, wp->cps_x[pts[i]], wp->cps_y[pts[i]], ((int) (0.06*imgSrcCp->width)));
			printf(" %f %f %f \n", wp->cps_x[pts[i]], wp->cps_y[pts[i]], diff);
			//cvCopy(imgSrc, imgWpd);
		}
		*/
		
		double smoo = 0;
		double diffX = 0;
		double diffY = 0;
		for (int i = 0; i<num_pts; i++) {
			diffX = midDspX - (wp->cps_x[pts[i]] - wp->dsp_x[pts[i]]);
			diffY = midDspY - (wp->cps_y[pts[i]] - wp->dsp_y[pts[i]]);
			smoo += sqrt(diffX*diffX + diffY*diffY);
		}
		printf(" %f %f \n \n", diff, smoo);
		/*
		wp->warpImg(imgSrcCp, imgWpdCp);
		wp->drawCpt(imgWpdCp);
		cvShowImage("Source", imgWpdCp);
		cvWaitKey(0);
		int a;
		*/
		//scanf(" %d ", &a);


		return diff+300*smoo;
    }

};

int main () {
	counter=0;

	/*
	warp *wp = new warp(11);
	wp->cps_x[50]-=0.05;
	wp->cps_y[50]+=0.05;
	*/

	if (SMP==1) {
		imgSrc = cvLoadImage( "src2.jpg" );
		imgTrg = cvLoadImage( "trg2.jpg" );
	}
	else {
		imgSrc = cvLoadImage( "src.jpg" );
		imgTrg = cvLoadImage( "trg.jpg" );
	}
	
	int width = imgSrc->width; 
	int height = imgSrc->height;
	

	/*
	cvNamedWindow( "1", CV_WINDOW_NORMAL );
	cvShowImage("1", imgSrc);
	cvNamedWindow( "2", CV_WINDOW_NORMAL );
	cvShowImage("2", imgTrg);
	*/
	

	imgWpd = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	
	//wp->drawCpt(imgWpd); // draw control points
	// stroka, stolbec:
	// cvSet2D(imgWpd,50,1,col);

	
	//wp->warpImg(imgSrc, imgWpd);

	cvCopy(imgSrc, imgWpd);
	cvNamedWindow( "Source", CV_WINDOW_NORMAL );
	cvResizeWindow("Source", width, height);
	cvShowImage("Source", imgWpd);

	// smallen
	
	imgSrcCp = cvCreateImage(cvSize(width / 5, height / 5), IPL_DEPTH_8U, 3);
	imgWpdCp = cvCreateImage(cvSize(width / 5, height / 5), IPL_DEPTH_8U, 3);
	imgTrgCp = cvCreateImage(cvSize(width / 5, height / 5), IPL_DEPTH_8U, 3);
	/*
	imgSrcCp = cvCreateImage(cvSize(width , height ), IPL_DEPTH_8U, 3);
	imgWpdCp = cvCreateImage(cvSize(width , height ), IPL_DEPTH_8U, 3);
	imgTrgCp = cvCreateImage(cvSize(width , height ), IPL_DEPTH_8U, 3);
	*/
	cvResize(imgSrc, imgSrcCp);
	cvCopy(imgSrcCp, imgWpdCp);
	cvResize(imgTrg, imgTrgCp);

	cvNamedWindow("Trg", CV_WINDOW_NORMAL);
	cvResizeWindow("Trg", width, height);
	cvShowImage("Trg", imgTrgCp);
	
	width /= 5;
	height /= 5;
	
	if (OPT==1) {
		int cps = 11;
		int rad = 0.06*width;
		//int num_pts = (cps + 1) * (cps + 1);
		wp = new warp(cps);
		bool inclNb = true; // include or not to include the neighbours
		double globalSSD = wp->warpedSSDfull(imgSrcCp, imgTrgCp); // ssd(imgSrcCp, imgTrgCp);
		printf("\n SSD: %f \n", globalSSD);
		test_function Func;

		for (int num = 0; num < 1; num++) {

			double tregion = 0.05;
			int cnt_stop = 0;

			if (num == 1) {
				width *= 5;
				height *= 5;
				rad *= 5;

				imgSrcCp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
				imgWpdCp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
				imgTrgCp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

				cvCopy(imgSrc, imgSrcCp);
				cvCopy(imgSrc, imgWpdCp);
				cvCopy(imgTrg, imgTrgCp);
			}

			while (cnt_stop<3) {
				int max_cp = wp->simpOptSg(imgWpdCp, imgTrgCp, rad);
				double max_dsp = (wp->dsp_x[max_cp] + wp->dsp_y[max_cp]);

				/*
				for (int i = 0; i< (wp->m + 1)*(wp->n + 1); i++) {
				if ((wp->dsp_x[i] + wp->dsp_y[i]) > THR1 * max_dsp) num_pts++;
				}
				*/

				if (!inclNb) {
					num_pts = 1; // just one point
					max_cp = 88;
				}
				else {
					num_pts = 0; // point with neighbourhood
					double diff_lim = 0.12 * 0.12;
					for (int i = 0; i < (wp->m + 1)*(wp->n + 1); i++) {
						double diffx = (wp->cps_x[i] - wp->cps_x[max_cp]);
						double diffy = (wp->cps_y[i] - wp->cps_y[max_cp]);
						double diff = diffx*diffx + diffy*diffy;
						if (diff < diff_lim) num_pts++;
					}
				}
				
				column_vector starting_point(2 * num_pts); // vector for BOBYQA
				column_vector low_point(2 * num_pts);
				column_vector hgh_point(2 * num_pts);
				
				pts = NULL; // indexes of points
				if (pts) free(pts);
				pts = (int *)malloc(num_pts*sizeof(int));
				memset(pts, 0, sizeof(pts));
				// filling
				num_pts = 0;

				
				if (!inclNb) { // just one point
					pts[num_pts] = max_cp;
					starting_point(2 * num_pts) = wp->cps_x[max_cp];
					low_point(2 * num_pts) = wp->cps_x[max_cp] - 0.055;
					hgh_point(2 * num_pts) = wp->cps_x[max_cp] + 0.055;
					starting_point(2 * num_pts + 1) = wp->cps_y[max_cp];
					low_point(2 * num_pts + 1) = wp->cps_y[max_cp] - 0.055;
					hgh_point(2 * num_pts + 1) = wp->cps_y[max_cp] + 0.055;
					num_pts++;
				}
				else {
					double diff_lim = 0.12 * 0.12;
					for (int i = 0; i < (wp->m + 1)*(wp->n + 1); i++) {
						double diffx = (wp->cps_x[i] - wp->cps_x[max_cp]);
						double diffy = (wp->cps_y[i] - wp->cps_y[max_cp]);
						double diff = diffx*diffx + diffy*diffy;

						if (diff < diff_lim) {
							pts[num_pts] = i;
							
							starting_point(2 * num_pts) = wp->cps_x[i];
							low_point(2 * num_pts) = wp->cps_x[i] - 0.055;
							hgh_point(2 * num_pts) = wp->cps_x[i] + 0.055;
							starting_point(2 * num_pts + 1) = wp->cps_y[i];
							low_point(2 * num_pts + 1) = wp->cps_y[i] - 0.055;
							hgh_point(2 * num_pts + 1) = wp->cps_y[i] + 0.055;
							
							num_pts++;
						}
					}
				}

				/*
				for (int i = 0; i< (wp->m + 1)*(wp->n + 1); i++) {
				if ((wp->dsp_x[i] + wp->dsp_y[i]) > THR1 * max_dsp) {
				pts[num_pts] = i;

				starting_point(2 * num_pts) = wp->cps_x[i];
				low_point(2 * num_pts) = wp->cps_x[i] - 0.07;
				hgh_point(2 * num_pts) = wp->cps_x[i] + 0.07;
				starting_point(2 * num_pts + 1) = wp->cps_y[i];
				low_point(2 * num_pts + 1) = wp->cps_y[i] - 0.07;
				hgh_point(2 * num_pts + 1) = wp->cps_y[i] + 0.07;

				num_pts++;

				}
				} */

				
				printf("Inputs: \n");
				for (int i = 0; i < num_pts; i++) {
					printf("%f %f \n", starting_point(2 * i), starting_point(2 * i + 1));
				}
				
				for (int i = 0; i < num_pts; i++) { // backup points
					wp->dsp_x[pts[i]] = wp->cps_x[pts[i]];
					wp->dsp_y[pts[i]] = wp->cps_y[pts[i]];
				}


					find_min_bobyqa(test_function(),
						starting_point,
						// (2 * num_pts + 1)*(2 * num_pts + 2) / 2,    //377 number of interpolation points
						(2 *num_pts + 1)*(2 *num_pts + 2) / 2,
						low_point,
						hgh_point,
						tregion,// initial trust region radius
						0.00001,  // stopping trust region radius 1e-6
						10000    // max number of objective function evaluations
						);

					/*
					uniform_matrix<double>(2 * num_pts, 1, 0),  // lower bound constraint
					uniform_matrix<double>(2 * num_pts, 1, 1),   // upper bound constraint
					*/
					
					printf("Results: \n");
					for (int i = 0; i < num_pts; i++) {
						printf("%f %f \n", starting_point(2 * i), starting_point(2 * i + 1));
					}
		
					for (int i = 0; i < num_pts; i++) {
						wp->cps_x[pts[i]] = starting_point(2 * i);
						wp->cps_y[pts[i]] = starting_point(2 * i + 1);
					}
	

					wp->warpImg(imgSrcCp, imgWpdCp);
					wp->drawCpt(imgWpdCp);
					cvShowImage("Source", imgWpdCp);
					double ssdNew = wp->warpedSSDfull(imgSrcCp, imgTrgCp);
					double locSSD = Func(starting_point);

					if ( (wp->localSSD[max_cp] != -1) && (locSSD < wp->localSSD[max_cp]) ) {
						wp->localSSD[max_cp] = locSSD;
					}
					else {
						wp->checked[max_cp] = true;
					}

					printf("\n SSD: new %f , old %f \n", ssdNew, globalSSD);

					if (ssdNew > 1.02 * globalSSD) {
						printf("backup: \n");
						for (int i = 0; i < num_pts; i++) {
							wp->cps_x[pts[i]] = wp->dsp_x[pts[i]];
							wp->cps_y[pts[i]] = wp->dsp_y[pts[i]];
							printf("%f %f \n", wp->cps_x[pts[i]], wp->cps_y[pts[i]]);
							//cvCopy(imgSrcCp, imgWpdCp);
						}
						cnt_stop++;
						//inclNb = true;
					}
					else {
						globalSSD = ssdNew;
						//tregion = 0.0001;
						//inclNb = false;
					}
	
					//cvWaitKey(0);
				
			}
		}

	}
	else {

		int rad = 30;
		int cps = 11;
		double ssd_val = ssd(imgSrc, imgTrg);

		while (rad > 2) {
			wp = new warp(cps);
			wp->simpOptSg(imgWpd, imgTrg, rad);
			printf(" %f %f ", wp->dsp_x[100], wp->dsp_y[100]);
			printf(" %f %f \n", wp->dsp_x[50], wp->dsp_y[50]);
			wp->warpImg(imgSrc, imgWpd);

			
			if (ssd(imgWpd, imgTrg)>ssd_val) {
				//cps*=2;
				//rad/=2;
			}
			

			ssd_val = ssd(imgWpd, imgTrg);
			cvCopy(imgWpd, imgSrc);

			printf("\n SSD: %f \n", ssd_val);
		
			wp->drawCpt(imgWpd);
			cvShowImage("Source", imgWpd);
			cvWaitKey(0);
			
		}
	}
	
	printf("\n SSD: %f \n", ssd(imgWpd, imgTrg));
	//wp->drawCpt(imgWpd);
	//wp2->drawCpt(imgSrc);

	//printf("SSD: %f %f \n", ssd_dir(imgSrc, imgTrg, 0.104067, 0.396732, 30, 1), ssd_dir(imgSrc, imgTrg, 0.104067, 0.396732, 30, 2));
	//printf("SSD: %f %f \n", ssd_dir(imgSrc, imgTrg, 0.104067, 0.396732, 30, 3), ssd_dir(imgSrc, imgTrg, 0.104067, 0.396732, 30, 4));
	// cycle on SSD's till minimizing

	wp->warpImg(imgSrc, imgWpd);
	wp->drawCpt(imgWpd);
    //cvShowImage("Source", imgSrc);
	cvShowImage("Source", imgWpd);
	cvResizeWindow("Source", width, height);


	

	cvWaitKey(0);
    
	cvDestroyWindow( "Example1" );
	//cvReleaseImage( &img );
	//cvReleaseImage( &imgWpd );
	
	//int a;
	//scanf(" %d ", &a);
	return 0;

}