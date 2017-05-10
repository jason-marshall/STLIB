// -*- C++ -*-

#if !defined(__numerical_random_PoissonGeneratorStochKit_ipp__)
#error This file is an implementation detail of PoissonGeneratorStochKit.
#endif

namespace stlib
{
namespace numerical {

template<class _Uniform, template<class> class Normal, typename _Result>
inline
typename PoissonGeneratorStochKit<_Uniform, Normal, _Result>::result_type
PoissonGeneratorStochKit<_Uniform, Normal, _Result>::
operator()(const argument_type mean) {
   // Call into ranlib for the Poisson deviate.
   if (mean == 0) {
      return 0;
   }
   else if (mean > 1e6) {
      return int(mean + 0.5);
   }
   else if (mean > 1e3) {
      // Round to the nearest integer.  We can reasonably assume it is positive.
      return int(mean + std::sqrt(mean)*snorm() + 0.5);
   }
   else {
      return ignpoi(float(mean));
   }
}





/*
**********************************************************************
int ignpoi(float mu)
GENerate POIsson random deviate
Function
Generates a single random deviate from a Poisson
distribution with mean AV.
Arguments
av --> The mean of the Poisson distribution from which
a random deviate is to be generated.
genexp <-- The random deviate.
Method
Renames KPOIS from TOMS as slightly modified by BWB to use RANF
instead of SUNIF.
For details see:
Ahrens, J.H. and Dieter, U.
Computer Generation of Poisson Generators
From Modified Normal Distributions.
ACM Trans. Math. Software, 8, 2
(June 1982),163-179
**********************************************************************
**********************************************************************


P O I S S O N  DISTRIBUTION


**********************************************************************
**********************************************************************

FOR DETAILS SEE:

AHRENS, J.H. AND DIETER, U.
COMPUTER GENERATION OF POISSON DEVIATES
FROM MODIFIED NORMAL DISTRIBUTIONS.
ACM TRANS. MATH. SOFTWARE, 8,2 (JUNE 1982), 163 - 179.

(SLIGHTLY MODIFIED VERSION OF THE PROGRAM IN THE ABOVE ARTICLE)

**********************************************************************
INTEGER FUNCTION IGNPOI(IR,MU)
INPUT:  IR=CURRENT STATE OF BASIC RANDOM NUMBER GENERATOR
MU=MEAN MU OF THE POISSON DISTRIBUTION
OUTPUT: IGNPOI=SAMPLE FROM THE POISSON-(MU)-DISTRIBUTION
MUPREV=PREVIOUS MU, MUOLD=MU AT LAST EXECUTION OF STEP P OR B.
TABLES: COEFFICIENTS A0-A7 FOR STEP F. FACTORIALS FACT
COEFFICIENTS A(K) - FOR PX = FK*V*V*SUM(A(K)*V**K)-DEL
SEPARATION OF CASES A AND B
*/
template<class _Uniform, template<class> class Normal, typename _Result>
inline
int
PoissonGeneratorStochKit<_Uniform, Normal, _Result>::
ignpoi(float mu) {
   //extern float fsign(float num, float sign);
   static float a0 = -0.5;
   static float a1 = 0.3333333;
   static float a2 = -0.2500068;
   static float a3 = 0.2000118;
   static float a4 = -0.1661269;
   static float a5 = 0.1421878;
   static float a6 = -0.1384794;
   static float a7 = 0.125006;
   static float muold = 0.0;
   static float muprev = 0.0;
   static float fact[10] = {
      1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0
   };
   static int ignpoi, j, k, kflag, l, m;
   static float b1, b2, c, c0, c1, c2, c3, d, del, difmuk, e, fk, fx, fy, g, omega, p, p0, px, py, q, s,
          t, u, v, x, xx, pp[35];

   if (mu == muprev) goto S10;
   if (mu < 10.0) goto S120;
   /*
     C A S E  A. (RECALCULATION OF S,D,L IF MU HAS CHANGED)
   */
   muprev = mu;
   s = sqrt(mu);
   d = 6.0 * mu * mu;
   /*
     THE POISSON PROBABILITIES PK EXCEED THE DISCRETE NORMAL
     PROBABILITIES FK WHENEVER K >= M(MU). L=IFIX(MU-1.1484)
     IS AN UPPER BOUND TO M(MU) FOR ALL MU >= 10 .
   */
   l = (int)(mu - 1.1484);
S10:
   /*
     STEP N. NORMAL SAMPLE - SNORM(IR) FOR STANDARD NORMAL DEVIATE
   */
   g = mu + s * snorm();
   if (g < 0.0) goto S20;
   ignpoi = (int)(g);
   /*
     STEP I. IMMEDIATE ACCEPTANCE IF IGNPOI IS LARGE ENOUGH
   */
   if (ignpoi >= l) return ignpoi;
   /*
     STEP S. SQUEEZE ACCEPTANCE - SUNIF(IR) FOR (0,1)-SAMPLE U
   */
   fk = (float)ignpoi;
   difmuk = mu - fk;
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   if (d* u >= difmuk*difmuk*difmuk) return ignpoi;
S20:
   /*
     STEP P. PREPARATIONS FOR STEPS Q AND H.
     (RECALCULATIONS OF PARAMETERS IF NECESSARY)
     .3989423=(2*PI)**(-.5)  .416667E-1=1./24.  .1428571=1./7.
     THE QUANTITIES B1, B2, C3, C2, C1, C0 ARE FOR THE HERMITE
     APPROXIMATIONS TO THE DISCRETE NORMAL PROBABILITIES FK.
     C=.1069/MU GUARANTEES MAJORIZATION BY THE 'HAT'-FUNCTION.
   */
   if (mu == muold) goto S30;
   muold = mu;
   omega = 0.3989423 / s;
   b1 = 4.166667E-2 / mu;
   b2 = 0.3 * b1 * b1;
   c3 = 0.1428571 * b1 * b2;
   c2 = b2 - 15.0 * c3;
   c1 = b1 - 6.0 * b2 + 45.0 * c3;
   c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3;
   c = 0.1069 / mu;
S30:
   if (g < 0.0) goto S50;
   /*
     'SUBROUTINE' F IS CALLED (KFLAG=0 FOR CORRECT RETURN)
   */
   kflag = 0;
   goto S70;
S40:
   /*
     STEP Q. QUOTIENT ACCEPTANCE (RARE CASE)
   */
   if (fy - u* fy <= py*std::exp(px - fx)) return ignpoi;
S50:
   /*
     STEP E. EXPONENTIAL SAMPLE - SEXPO(IR) FOR STANDARD EXPONENTIAL
     DEVIATE E AND SAMPLE T FROM THE LAPLACE 'HAT'
     (IF T <= -.6744 THEN PK < FK FOR ALL MU >= 10.)
   */
   e = sexpo();
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   u += (u - 1.0);
   t = 1.8 + fsign(e, u);
   if (t <= -0.6744) goto S50;
   ignpoi = (int)(mu + s * t);
   fk = (float)ignpoi;
   difmuk = mu - fk;
   /*
     'SUBROUTINE' F IS CALLED (KFLAG=1 FOR CORRECT RETURN)
   */
   kflag = 1;
   goto S70;
S60:
   /*
     STEP H. HAT ACCEPTANCE (E IS REPEATED ON REJECTION)
   */
   if (c*std::abs(u) > py*std::exp(px + e) - fy*std::exp(fx + e)) goto S50;
   return ignpoi;
S70:
   /*
     STEP F. 'SUBROUTINE' F. CALCULATION OF PX,PY,FX,FY.
     CASE IGNPOI .LT. 10 USES FACTORIALS FROM TABLE FACT
   */
   if (ignpoi >= 10) goto S80;
   px = -mu;
   py = pow(mu, (double)ignpoi) / *(fact + ignpoi);
   goto S110;
S80:
   /*
     CASE IGNPOI .GE. 10 USES POLYNOMIAL APPROXIMATION
     A0-A7 FOR ACCURACY WHEN ADVISABLE
     .8333333E-1=1./12.  .3989423=(2*PI)**(-.5)
   */
   del = 8.333333E-2 / fk;
   del -= (4.8 * del * del * del);
   v = difmuk / fk;
   if (std::abs(v) <= 0.25) goto S90;
   px = fk * std::log(1.0 + v) - difmuk - del;
   goto S100;
S90:
   px = fk * v * v * (((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v + a0) - del;
S100:
   py = 0.3989423 / sqrt(fk);
S110:
   x = (0.5 - difmuk) / s;
   xx = x * x;
   fx = -0.5 * xx;
   fy = omega * (((c3 * xx + c2) * xx + c1) * xx + c0);
   if (kflag <= 0) goto S40;
   goto S60;
S120:
   /*
     C A S E  B. (START NEW TABLE AND CALCULATE P0 IF NECESSARY)
   */
   muprev = 0.0;
   if (mu == muold) goto S130;
   muold = mu;
   m = std::max(1, (int)(mu));
   l = 0;
   p = std::exp(-mu);
   q = p0 = p;
S130:
   /*
     STEP U. UNIFORM SAMPLE FOR INVERSION METHOD
   */
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   ignpoi = 0;
   if (u <= p0) return ignpoi;
   /*
     STEP T. TABLE COMPARISON UNTIL THE END PP(L) OF THE
     PP-TABLE OF CUMULATIVE POISSON PROBABILITIES
     (0.458=PP(9) FOR MU=10)
   */
   if (l == 0) goto S150;
   j = 1;
   if (u > 0.458) j = std::min(l, m);
   for (k = j; k <= l; k++) {
      if (u <= *(pp + k - 1)) goto S180;
   }
   if (l == 35) goto S130;
S150:
   /*
     STEP C. CREATION OF NEW POISSON PROBABILITIES P
     AND THEIR CUMULATIVES Q=PP(K)
   */
   l += 1;
   for (k = l; k <= 35; k++) {
      p = p * mu / (float)k;
      q += p;
      *(pp + k - 1) = q;
      if (u <= q) goto S170;
   }
   l = 35;
   goto S130;
S170:
   l = k;
S180:
   ignpoi = k;
   return ignpoi;
}



/*
**********************************************************************


(STANDARD-)  N O R M A L  DISTRIBUTION


**********************************************************************
**********************************************************************

FOR DETAILS SEE:

AHRENS, J.H. AND DIETER, U.
EXTENSIONS OF FORSYTHE'S METHOD FOR RANDOM
SAMPLING FROM THE NORMAL DISTRIBUTION.
MATH. COMPUT., 27,124 (OCT. 1973), 927 - 937.

ALL STATEMENT NUMBERS CORRESPOND TO THE STEPS OF ALGORITHM 'FL'
(M=5) IN THE ABOVE PAPER     (SLIGHTLY MODIFIED IMPLEMENTATION)

Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of
SUNIF.  The argument IR thus goes away.

**********************************************************************
THE DEFINITIONS OF THE CONSTANTS A(K), D(K), T(K) AND
H(K) ARE ACCORDING TO THE ABOVEMENTIONED ARTICLE
*/
template<class _Uniform, template<class> class Normal, typename _Result>
inline
float
PoissonGeneratorStochKit<_Uniform, Normal, _Result>::
snorm() {
   static float a[32] = {
      0.0, 3.917609E-2, 7.841241E-2, 0.11777, 0.1573107, 0.1970991, 0.2372021, 0.2776904,
      0.3186394, 0.36013, 0.4022501, 0.4450965, 0.4887764, 0.5334097, 0.5791322,
      0.626099, 0.6744898, 0.7245144, 0.7764218, 0.8305109, 0.8871466, 0.9467818,
      1.00999, 1.077516, 1.150349, 1.229859, 1.318011, 1.417797, 1.534121, 1.67594,
      1.862732, 2.153875
   };
   static float d[31] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.2636843, 0.2425085, 0.2255674, 0.2116342, 0.1999243,
      0.1899108, 0.1812252, 0.1736014, 0.1668419, 0.1607967, 0.1553497, 0.1504094,
      0.1459026, 0.14177, 0.1379632, 0.1344418, 0.1311722, 0.128126, 0.1252791,
      0.1226109, 0.1201036, 0.1177417, 0.1155119, 0.1134023, 0.1114027, 0.1095039
   };
   static float t[31] = {
      7.673828E-4, 2.30687E-3, 3.860618E-3, 5.438454E-3, 7.0507E-3, 8.708396E-3,
      1.042357E-2, 1.220953E-2, 1.408125E-2, 1.605579E-2, 1.81529E-2, 2.039573E-2,
      2.281177E-2, 2.543407E-2, 2.830296E-2, 3.146822E-2, 3.499233E-2, 3.895483E-2,
      4.345878E-2, 4.864035E-2, 5.468334E-2, 6.184222E-2, 7.047983E-2, 8.113195E-2,
      9.462444E-2, 0.1123001, 0.136498, 0.1716886, 0.2276241, 0.330498, 0.5847031
   };
   static float h[31] = {
      3.920617E-2, 3.932705E-2, 3.951E-2, 3.975703E-2, 4.007093E-2, 4.045533E-2,
      4.091481E-2, 4.145507E-2, 4.208311E-2, 4.280748E-2, 4.363863E-2, 4.458932E-2,
      4.567523E-2, 4.691571E-2, 4.833487E-2, 4.996298E-2, 5.183859E-2, 5.401138E-2,
      5.654656E-2, 5.95313E-2, 6.308489E-2, 6.737503E-2, 7.264544E-2, 7.926471E-2,
      8.781922E-2, 9.930398E-2, 0.11556, 0.1404344, 0.1836142, 0.2790016, 0.7010474
   };
   static int i;
   static float snorm, u, s, ustar, aa, w, y, tt;
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   s = 0.0;
   if (u > 0.5) s = 1.0;
   u += (u - s);
   u = 32.0 * u;
   i = (int)(u);
   if (i == 32) i = 31;
   if (i == 0) goto S100;
   /*
     START CENTER
   */
   ustar = u - (float)i;
   aa = *(a + i - 1);
S40:
   if (ustar <= *(t + i - 1)) goto S60;
   w = (ustar - *(t + i - 1))** (h + i - 1);
S50:
   /*
     EXIT   (BOTH CASES)
   */
   y = aa + w;
   snorm = y;
   if (s == 1.0) snorm = -y;
   return snorm;
S60:
   /*
     CENTER CONTINUED
   */
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   w = u * (*(a + i) - aa);
   tt = (0.5 * w + aa) * w;
   goto S80;
S70:
   tt = u;
   ustar = transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_normalGenerator->getDiscreteUniformGenerator())());
S80:
   if (ustar > tt) goto S50;
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   if (ustar >= u) goto S70;
   ustar = transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_normalGenerator->getDiscreteUniformGenerator())());
   goto S40;
S100:
   /*
     START TAIL
   */
   i = 6;
   aa = *(a + 31);
   goto S120;
S110:
   aa += *(d + i - 1);
   i += 1;
S120:
   u += u;
   if (u < 1.0) goto S110;
   u -= 1.0;
S140:
   w = u** (d + i - 1);
   tt = (0.5 * w + aa) * w;
   goto S160;
S150:
   tt = u;
S160:
   ustar = transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_normalGenerator->getDiscreteUniformGenerator())());
   if (ustar > tt) goto S50;
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   if (ustar >= u) goto S150;
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   goto S140;
}


/*
**********************************************************************


(STANDARD-)  E X P O N E N T I A L   DISTRIBUTION


**********************************************************************
**********************************************************************

FOR DETAILS SEE:

AHRENS, J.H. AND DIETER, U.
COMPUTER METHODS FOR SAMPLING FROM THE
EXPONENTIAL AND NORMAL DISTRIBUTIONS.
COMM. ACM, 15,10 (OCT. 1972), 873 - 882.

ALL STATEMENT NUMBERS CORRESPOND TO THE STEPS OF ALGORITHM
'SA' IN THE ABOVE PAPER (SLIGHTLY MODIFIED IMPLEMENTATION)

Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of
SUNIF.  The argument IR thus goes away.

**********************************************************************
Q(N) = SUM(ALOG(2.0)**K/K!)    K=1,..,N ,      THE HIGHEST N
(HERE 8) IS DETERMINED BY Q(N)=1.0 WITHIN STANDARD PRECISION
*/
template<class _Uniform, template<class> class Normal, typename _Result>
inline
float
PoissonGeneratorStochKit<_Uniform, Normal, _Result>::
sexpo() {
   static float q[8] = {
      0.6931472, 0.9333737, 0.9888778, 0.9984959, 0.9998293, 0.9999833, 0.9999986, 1.0
   };
   static int i;
   static float sexpo, a, u, ustar, umin;
   static float* q1 = q;
   a = 0.0;
   u = transformDiscreteDeviateToContinuousDeviateClosed<Number>
       ((*_normalGenerator->getDiscreteUniformGenerator())());
   goto S30;
S20:
   a += *q1;
S30:
   u += u;
   if (u <= 1.0) goto S20;
   u -= 1.0;
   if (u > *q1) goto S60;
   sexpo = a + u;
   return sexpo;
S60:
   i = 1;
   ustar = transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_normalGenerator->getDiscreteUniformGenerator())());
   umin = ustar;
S70:
   ustar = transformDiscreteDeviateToContinuousDeviateClosed<Number>
           ((*_normalGenerator->getDiscreteUniformGenerator())());
   if (ustar < umin) umin = ustar;
   i += 1;
   if (u > *(q + i - 1)) goto S70;
   sexpo = a + umin** q1;
   return sexpo;
}




/* Transfers sign of argument sign to argument num */
template<class _Uniform, template<class> class Normal, typename _Result>
inline
float
PoissonGeneratorStochKit<_Uniform, Normal, _Result>::
fsign(float num, float sign) {
   if ((sign > 0.0f && num < 0.0f) || (sign<0.0f && num>0.0f))
      return -num;
   else return num;
}

} // namespace numerical
}
