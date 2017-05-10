// -*- C++ -*-

#if !defined(__geom_mesh_iss_quality_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


// Calculate the adjacency counts for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
countAdjacencies(const IndSimpSetIncAdj<N, M, T>& iss,
                 std::array < std::size_t, M + 2 > * counts) {
   std::fill(counts->begin(), counts->end(), 0);
   // For each simplex.
   for (std::size_t n = 0; n != iss.indexedSimplices.size(); ++n) {
      ++(*counts)[numAdjacent(iss.adjacent[n])];
   }
}



// Identify the elements that have a low quality using the condition number metric.
template<std::size_t N, std::size_t M, typename T, typename OutputIterator>
inline
void
identifyLowQualityWithCondNum(const IndSimpSet<N, M, T>& mesh,
                              const T minimumAllowedQuality,
                              OutputIterator indices) {
   typedef typename IndSimpSet<N, M, T>::Simplex Simplex;

   assert(0 < minimumAllowedQuality && minimumAllowedQuality <= 1);
   const T maximumDistortion = 1. / minimumAllowedQuality;

   Simplex s;
   SimplexCondNum<M, T> f;
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      mesh.getSimplex(i, &s);
      f.setFunction(s);
      if (f.getDeterminant() <= 0 || f() > maximumDistortion) {
         *indices++ = i;
      }
   }
}

// Compute edge length statistics.
// With a bucket of simplices, one can compute the minimum and maximum, but
// not the mean.
template<std::size_t M, typename SimpInIter, typename T>
inline
void
computeEdgeLengthStatistics(SimpInIter simplicesBeginning,
                            SimpInIter simplicesEnd,
                            T* minimumLength,
                            T* maximumLength) {
   *minimumLength = std::numeric_limits<T>::max();
   *maximumLength = -std::numeric_limits<T>::max();
   T d;
   // For each simplex.
   for (; simplicesBeginning != simplicesEnd; ++simplicesBeginning) {
      // For each edge (pair of vertices).
      for (std::size_t i = 0; i != M; ++i) {
         for (std::size_t j = i + 1; j != M + 1; ++j) {
            d = ext::euclideanDistance((*simplicesBeginning)[i],
                                       (*simplicesBeginning)[j]);
            if (d < *minimumLength) {
               *minimumLength = d;
            }
            if (d > *maximumLength) {
               *maximumLength = d;
            }
         }
      }
   }
}



// Compute edge length statistics.
template<std::size_t M, typename VertRAIter, typename ISInIter, typename T>
void
computeEdgeLengthStatistics(VertRAIter vertices,
                            ISInIter indexedSimplicesBeginning,
                            ISInIter indexedSimplicesEnd,
                            T* minimumLength,
                            T* maximumLength) {
   *minimumLength = std::numeric_limits<T>::max();
   *maximumLength = -std::numeric_limits<T>::max();
   T d;
   // For each simplex.
   for (; indexedSimplicesBeginning != indexedSimplicesEnd;
         ++indexedSimplicesBeginning) {
      // For each edge (pair of vertices).
      for (std::size_t i = 0; i != M; ++i) {
         for (std::size_t j = i + 1; j != M + 1; ++j) {
            d = ext::euclideanDistance
              (vertices[(*indexedSimplicesBeginning)[i]],
               vertices[(*indexedSimplicesBeginning)[j]]);
            if (d < *minimumLength) {
               *minimumLength = d;
            }
            if (d > *maximumLength) {
               *maximumLength = d;
            }
         }
      }
   }
}



// Compute edge length statistics.
// CONTINUE: Add a edge iterator to IndSimpSetIncAdj.  Then I could easily
// implement this for M = 3.
template<std::size_t N, typename T>
inline
void
computeEdgeLengthStatistics(const IndSimpSetIncAdj<N, 2, T>& mesh,
                            T* minimumLength,
                            T* maximumLength,
                            T* meanLength) {
   typedef typename IndSimpSetIncAdj<N, 2, T>::FaceIterator FaceIterator;

   const std::size_t M = 2;

   // Initialize the statistics.
   *minimumLength = std::numeric_limits<T>::max();
   *maximumLength = -std::numeric_limits<T>::max();
   *meanLength = 0;

   std::size_t numberOfEdges = 0;
   std::size_t simplexIndex, localIndex, vertexIndex1, vertexIndex2;
   T length;
   // For each face (edge).
   const FaceIterator iEnd = mesh.getFacesEnd();
   for (FaceIterator i = mesh.getFacesBeginning(); i != iEnd; ++i) {
      ++numberOfEdges;
      simplexIndex = i->first;
      localIndex = i->second;
      vertexIndex1 = mesh.indexedSimplices[simplexIndex][(localIndex + 1) %
                     (M + 1)];
      vertexIndex2 = mesh.indexedSimplices[simplexIndex][(localIndex + 2) %
                     (M + 1)];
      length = ext::euclideanDistance(mesh.vertices[vertexIndex1],
                                 mesh.vertices[vertexIndex2]);
      if (length < *minimumLength) {
         *minimumLength = length;
      }
      if (length > *maximumLength) {
         *maximumLength = length;
      }
      *meanLength += length;
   }
   if (numberOfEdges != 0) {
      *meanLength /= numberOfEdges;
   }
}



// Return the minimum edge length.
template<std::size_t M, typename T, typename SimpInIter>
inline
T
computeMinimumEdgeLength(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd) {
   T minimumLength, maximumLength;
   computeEdgeLengthStatistics<M>(simplicesBeginning, simplicesEnd,
                                  &minimumLength, &maximumLength);
   return minimumLength;
}



// Return the maximum edge length.
template<std::size_t M, typename T, typename SimpInIter>
inline
T
computeMaximumEdgeLength(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd) {
   T minimumLength, maximumLength;
   computeEdgeLengthStatistics<M>(simplicesBeginning, simplicesEnd,
                                  &minimumLength, &maximumLength);
   return maximumLength;
}



// Return the total content of the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
inline
T
computeContent(VertRAIter vertices,
               ISInIter indexedSimplicesBeginning,
               ISInIter indexedSimplicesEnd) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   Simplex s;
   SimplexJac<M, T> sj;
   T c = 0;
   // Loop over the simplices.
   for (; indexedSimplicesBeginning != indexedSimplicesEnd;
         ++indexedSimplicesBeginning) {
      for (std::size_t i = 0; i != s.size(); ++i) {
         s[i] = vertices[(*indexedSimplicesBeginning)[i]];
      }
      sj.setFunction(s);
      c += sj.computeContent();
   }
   return c;
}


// Return the total content of the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
inline
T
computeContent(SimpInIter simplicesBeginning, SimpInIter simplicesEnd) {
   SimplexJac<M, T> sj;
   T c = 0;
   // Loop over the simplices.
   for (; simplicesBeginning != simplicesEnd; ++simplicesBeginning) {
      sj.setFunction(*simplicesBeginning);
      c += sj.computeContent();
   }
   return c;
}


// Return the total content of the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
T
computeContent(const IndSimpSet<N, M, T>& iss) {
   return computeContent<M, T>(iss.vertices.begin(),
                               iss.indexedSimplices.begin(),
                               iss.indexedSimplices.end());
}



// Calculate content (hypervolume) statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
inline
void
computeContentStatistics(VertRAIter vertices,
                         ISInIter indexedSimplicesBeginning,
                         ISInIter indexedSimplicesEnd,
                         T* minContent,
                         T* maxContent,
                         T* meanContent) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   T minCont = std::numeric_limits<T>::max();
   T maxCont = -std::numeric_limits<T>::max();
   T sumCont = 0;
   T x;

   Simplex s;
   SimplexJac<M, T> sj;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; indexedSimplicesBeginning != indexedSimplicesEnd;
         ++indexedSimplicesBeginning, ++numSimplices) {
      for (std::size_t i = 0; i != s.size(); ++i) {
         s[i] = vertices[(*indexedSimplicesBeginning)[i]];
      }
      sj.setFunction(s);
      x = sj.computeContent();
      if (x < minCont) {
         minCont = x;
      }
      if (x > maxCont) {
         maxCont = x;
      }
      sumCont += x;
   }

   if (numSimplices == 0) {
      *minContent = 0;
      *maxContent = 0;
      *meanContent = 0;
   }
   else {
      *minContent = minCont;
      *maxContent = maxCont;
      *meanContent = sumCont / numSimplices;
   }
}



// Calculate content (hypervolume) statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
inline
void
computeContentStatistics(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd,
                         T* minContent,
                         T* maxContent,
                         T* meanContent) {
   T minCont = std::numeric_limits<T>::max();
   T maxCont = -std::numeric_limits<T>::max();
   T sumCont = 0;
   T x;

   SimplexJac<M, T> sj;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; simplicesBeginning != simplicesEnd;
         ++simplicesBeginning, ++numSimplices) {
      sj.setFunction(*simplicesBeginning);
      x = sj.computeContent();
      if (x < minCont) {
         minCont = x;
      }
      if (x > maxCont) {
         maxCont = x;
      }
      sumCont += x;
   }

   if (numSimplices == 0) {
      *minContent = 0;
      *maxContent = 0;
      *meanContent = 0;
   }
   else {
      *minContent = minCont;
      *maxContent = maxCont;
      *meanContent = sumCont / numSimplices;
   }
}



// Calculate content (hypervolume) statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
computeContentStatistics(const IndSimpSet<N, M, T>& iss,
                         T* minContent,
                         T* maxContent,
                         T* meanContent) {
   computeContentStatistics<M>(iss.vertices.begin(),
                               iss.indexedSimplices.begin(),
                               iss.indexedSimplices.end(),
                               minContent, maxContent, meanContent);
}




// Calculate determinant statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
inline
void
computeDeterminantStatistics(VertRAIter vertices,
                             ISInIter indexedSimplicesBeginning,
                             ISInIter indexedSimplicesEnd,
                             T* minDeterminant,
                             T* maxDeterminant,
                             T* meanDeterminant) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   T minDet = std::numeric_limits<T>::max();
   T maxDet = -std::numeric_limits<T>::max();
   T sumDet = 0;
   T x;

   Simplex s;
   SimplexJac<M, T> sj;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; indexedSimplicesBeginning != indexedSimplicesEnd;
         ++indexedSimplicesBeginning, ++numSimplices) {
      for (std::size_t i = 0; i != s.size(); ++i) {
         s[i] = vertices[(*indexedSimplicesBeginning)[i]];
      }
      sj.setFunction(s);
      x = sj.getDeterminant();
      if (x < minDet) {
         minDet = x;
      }
      if (x > maxDet) {
         maxDet = x;
      }
      sumDet += x;
   }

   if (numSimplices == 0) {
      *minDeterminant = 0;
      *maxDeterminant = 0;
      *meanDeterminant = 0;
   }
   else {
      *minDeterminant = minDet;
      *maxDeterminant = maxDet;
      *meanDeterminant = sumDet / numSimplices;
   }
}



// Calculate determinant statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
inline
void
computeDeterminantStatistics(SimpInIter simplicesBeginning,
                             SimpInIter simplicesEnd,
                             T* minDeterminant,
                             T* maxDeterminant,
                             T* meanDeterminant) {
   T minDet = std::numeric_limits<T>::max();
   T maxDet = -std::numeric_limits<T>::max();
   T sumDet = 0;
   T x;

   SimplexJac<M, T> sj;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; simplicesBeginning != simplicesEnd;
         ++simplicesBeginning, ++numSimplices) {
      sj.setFunction(*simplicesBeginning);
      x = sj.getDeterminant();
      if (x < minDet) {
         minDet = x;
      }
      if (x > maxDet) {
         maxDet = x;
      }
      sumDet += x;
   }

   if (numSimplices == 0) {
      *minDeterminant = 0;
      *maxDeterminant = 0;
      *meanDeterminant = 0;
   }
   else {
      *minDeterminant = minDet;
      *maxDeterminant = maxDet;
      *meanDeterminant = sumDet / numSimplices;
   }
}



// Calculate determinant statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
computeDeterminantStatistics(const IndSimpSet<N, M, T>& iss,
                             T* minDeterminant,
                             T* maxDeterminant,
                             T* meanDeterminant) {
   computeDeterminantStatistics<M>(iss.vertices.begin(),
                                   iss.indexedSimplices.begin(),
                                   iss.indexedSimplices.end(),
                                   minDeterminant, maxDeterminant,
                                   meanDeterminant);
}






// Calculate modified mean ratio function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
inline
void
computeModifiedMeanRatioStatistics(VertRAIter vertices,
                                   ISInIter indexedSimplicesBeginning,
                                   ISInIter indexedSimplicesEnd,
                                   T* minModMeanRatio,
                                   T* maxModMeanRatio,
                                   T* meanModMeanRatio) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   T minMmr = std::numeric_limits<T>::max();
   T maxMmr = -std::numeric_limits<T>::max();
   T sumMmr = 0;
   T x;

   Simplex s;
   SimplexModMeanRatio<M, T> smmr;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; indexedSimplicesBeginning != indexedSimplicesEnd;
         ++indexedSimplicesBeginning, ++numSimplices) {
      for (std::size_t i = 0; i != s.size(); ++i) {
         s[i] = vertices[(*indexedSimplicesBeginning)[i]];
      }
      smmr.setFunction(s);
      x = smmr();
      if (x < minMmr) {
         minMmr = x;
      }
      if (x > maxMmr) {
         maxMmr = x;
      }
      sumMmr += x;
   }

   if (numSimplices == 0) {
      *minModMeanRatio = 0;
      *maxModMeanRatio = 0;
      *meanModMeanRatio = 0;
   }
   else {
      *minModMeanRatio = minMmr;
      *maxModMeanRatio = maxMmr;
      *meanModMeanRatio = sumMmr / numSimplices;
   }
}


// Calculate modified mean ratio function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
inline
void
computeModifiedMeanRatioStatistics(SimpInIter simplicesBeginning,
                                   SimpInIter simplicesEnd,
                                   T* minModMeanRatio,
                                   T* maxModMeanRatio,
                                   T* meanModMeanRatio) {
   T minMmr = std::numeric_limits<T>::max();
   T maxMmr = -std::numeric_limits<T>::max();
   T sumMmr = 0;
   T x;

   SimplexModMeanRatio<M, T> smmr;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; simplicesBeginning != simplicesEnd;
         ++simplicesBeginning, ++numSimplices) {
      smmr.setFunction(*simplicesBeginning);
      x = smmr();
      if (x < minMmr) {
         minMmr = x;
      }
      if (x > maxMmr) {
         maxMmr = x;
      }
      sumMmr += x;
   }

   if (numSimplices == 0) {
      *minModMeanRatio = 0;
      *maxModMeanRatio = 0;
      *meanModMeanRatio = 0;
   }
   else {
      *minModMeanRatio = minMmr;
      *maxModMeanRatio = maxMmr;
      *meanModMeanRatio = sumMmr / numSimplices;
   }
}


// Calculate modified mean ratio function statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
computeModifiedMeanRatioStatistics(const IndSimpSet<N, M, T>& iss,
                                   T* minModMeanRatio,
                                   T* maxModMeanRatio,
                                   T* meanModMeanRatio) {
   computeModifiedMeanRatioStatistics<M>(iss.vertices.begin(),
                                         iss.indexedSimplices.begin(),
                                         iss.indexedSimplices.end(),
                                         minModMeanRatio, maxModMeanRatio,
                                         meanModMeanRatio);
}







// Calculate modified condition number function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
inline
void
computeModifiedConditionNumberStatistics(VertRAIter vertices,
      ISInIter indexedSimplicesBeginning,
      ISInIter indexedSimplicesEnd,
      T* minModCondNum,
      T* maxModCondNum,
      T* meanModCondNum) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   T minMcn = std::numeric_limits<T>::max();
   T maxMcn = -std::numeric_limits<T>::max();
   T sumMcn = 0;
   T x;

   Simplex s;
   SimplexModCondNum<M, T> smcn;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; indexedSimplicesBeginning != indexedSimplicesEnd; ++indexedSimplicesBeginning, ++numSimplices) {
      for (std::size_t i = 0; i != s.size(); ++i) {
         s[i] = vertices[(*indexedSimplicesBeginning)[i]];
      }
      smcn.setFunction(s);
      x = smcn();
      if (x < minMcn) {
         minMcn = x;
      }
      if (x > maxMcn) {
         maxMcn = x;
      }
      sumMcn += x;
   }

   if (numSimplices == 0) {
      *minModCondNum = 0;
      *maxModCondNum = 0;
      *meanModCondNum = 0;
   }
   else {
      *minModCondNum = minMcn;
      *maxModCondNum = maxMcn;
      *meanModCondNum = sumMcn / numSimplices;
   }
}


// Calculate modified condition number function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
inline
void
computeModifiedConditionNumberStatistics(SimpInIter simplicesBeginning,
      SimpInIter simplicesEnd,
      T* minModCondNum,
      T* maxModCondNum,
      T* meanModCondNum) {
   T minMcn = std::numeric_limits<T>::max();
   T maxMcn = -std::numeric_limits<T>::max();
   T sumMcn = 0;
   T x;

   SimplexModCondNum<M, T> smcn;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; simplicesBeginning != simplicesEnd;
         ++simplicesBeginning, ++numSimplices) {
      smcn.setFunction(*simplicesBeginning);
      x = smcn();
      if (x < minMcn) {
         minMcn = x;
      }
      if (x > maxMcn) {
         maxMcn = x;
      }
      sumMcn += x;
   }

   if (numSimplices == 0) {
      *minModCondNum = 0;
      *maxModCondNum = 0;
      *meanModCondNum = 0;
   }
   else {
      *minModCondNum = minMcn;
      *maxModCondNum = maxMcn;
      *meanModCondNum = sumMcn / numSimplices;
   }
}


// Calculate modified condition number function statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
computeModifiedConditionNumberStatistics(const IndSimpSet<N, M, T>& iss,
      T* minModCondNum,
      T* maxModCondNum,
      T* meanModCondNum) {
   computeModifiedConditionNumberStatistics<M>
   (iss.vertices.begin(),
    iss.indexedSimplices.begin(),
    iss.indexedSimplices.end(),
    minModCondNum, maxModCondNum, meanModCondNum);
}





// Calculate quality statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
inline
void
computeQualityStatistics(VertRAIter vertices,
                         ISInIter indexedSimplicesBeginning,
                         ISInIter indexedSimplicesEnd,
                         T* minContent,
                         T* maxContent,
                         T* meanContent,
                         T* minDeterminant,
                         T* maxDeterminant,
                         T* meanDeterminant,
                         T* minModMeanRatio,
                         T* maxModMeanRatio,
                         T* meanModMeanRatio,
                         T* minModCondNum,
                         T* maxModCondNum,
                         T* meanModCondNum) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   T minCont = std::numeric_limits<T>::max();
   T maxCont = -std::numeric_limits<T>::max();
   T sumCont = 0;
   T minDet = std::numeric_limits<T>::max();
   T maxDet = -std::numeric_limits<T>::max();
   T sumDet = 0;
   T minMmr = std::numeric_limits<T>::max();
   T maxMmr = -std::numeric_limits<T>::max();
   T sumMmr = 0;
   T minMcn = std::numeric_limits<T>::max();
   T maxMcn = -std::numeric_limits<T>::max();
   T sumMcn = 0;
   T x;

   Simplex s;
   SimplexModMeanRatio<M, T> smmr;
   SimplexModCondNum<M, T> smcn;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; indexedSimplicesBeginning != indexedSimplicesEnd;
         ++indexedSimplicesBeginning, ++numSimplices) {
      for (std::size_t i = 0; i != s.size(); ++i) {
         s[i] = vertices[(*indexedSimplicesBeginning)[i]];
      }
      smmr.setFunction(s);
      smcn.setFunction(s);
      x = smmr.computeContent();
      if (x < minCont) {
         minCont = x;
      }
      if (x > maxCont) {
         maxCont = x;
      }
      sumCont += x;

      x = smmr.getDeterminant();
      if (x < minDet) {
         minDet = x;
      }
      if (x > maxDet) {
         maxDet = x;
      }
      sumDet += x;

      x = 1.0 / smmr();
      if (x < minMmr) {
         minMmr = x;
      }
      if (x > maxMmr) {
         maxMmr = x;
      }
      sumMmr += x;

      x = 1.0 / smcn();
      if (x < minMcn) {
         minMcn = x;
      }
      if (x > maxMcn) {
         maxMcn = x;
      }
      sumMcn += x;
   }

   if (numSimplices == 0) {
      *minContent = 0;
      *maxContent = 0;
      *meanContent = 0;
      *minDeterminant = 0;
      *maxDeterminant = 0;
      *meanDeterminant = 0;
      *minModMeanRatio = 0;
      *maxModMeanRatio = 0;
      *meanModMeanRatio = 0;
      *minModCondNum = 0;
      *maxModCondNum = 0;
      *meanModCondNum = 0;
   }
   else {
      *minContent = minCont;
      *maxContent = maxCont;
      *meanContent = sumCont / numSimplices;
      *minDeterminant = minDet;
      *maxDeterminant = maxDet;
      *meanDeterminant = sumDet / numSimplices;
      *minModMeanRatio = minMmr;
      *maxModMeanRatio = maxMmr;
      *meanModMeanRatio = sumMmr / numSimplices;
      *minModCondNum = minMcn;
      *maxModCondNum = maxMcn;
      *meanModCondNum = sumMcn / numSimplices;
   }
}


// Calculate quality statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
inline
void
computeQualityStatistics(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd,
                         T* minContent,
                         T* maxContent,
                         T* meanContent,
                         T* minDeterminant,
                         T* maxDeterminant,
                         T* meanDeterminant,
                         T* minModMeanRatio,
                         T* maxModMeanRatio,
                         T* meanModMeanRatio,
                         T* minModCondNum,
                         T* maxModCondNum,
                         T* meanModCondNum) {
   typedef typename std::iterator_traits<SimpInIter>::value_type Value;
   typedef typename 
     std::remove_const<typename std::remove_volatile<Value>::type>::type
     Simplex;

   T minCont = std::numeric_limits<T>::max();
   T maxCont = -std::numeric_limits<T>::max();
   T sumCont = 0;
   T minDet = std::numeric_limits<T>::max();
   T maxDet = -std::numeric_limits<T>::max();
   T sumDet = 0;
   T minMmr = std::numeric_limits<T>::max();
   T maxMmr = -std::numeric_limits<T>::max();
   T sumMmr = 0;
   T minMcn = std::numeric_limits<T>::max();
   T maxMcn = -std::numeric_limits<T>::max();
   T sumMcn = 0;
   T x;

   Simplex s;
   SimplexModMeanRatio<M, T> smmr;
   SimplexModCondNum<M, T> smcn;
   // Loop over the simplices.
   std::size_t numSimplices = 0;
   for (; simplicesBeginning != simplicesEnd; ++simplicesBeginning, ++numSimplices) {
      s = *simplicesBeginning;
      smmr.setFunction(s);
      smcn.setFunction(s);
      x = smmr.computeContent();
      if (x < minCont) {
         minCont = x;
      }
      if (x > maxCont) {
         maxCont = x;
      }
      sumCont += x;

      x = smmr.getDeterminant();
      if (x < minDet) {
         minDet = x;
      }
      if (x > maxDet) {
         maxDet = x;
      }
      sumDet += x;

      x = 1.0 / smmr();
      if (x < minMmr) {
         minMmr = x;
      }
      if (x > maxMmr) {
         maxMmr = x;
      }
      sumMmr += x;

      x = 1.0 / smcn();
      if (x < minMcn) {
         minMcn = x;
      }
      if (x > maxMcn) {
         maxMcn = x;
      }
      sumMcn += x;
   }

   if (numSimplices == 0) {
      *minContent = 0;
      *maxContent = 0;
      *meanContent = 0;
      *minDeterminant = 0;
      *maxDeterminant = 0;
      *meanDeterminant = 0;
      *minModMeanRatio = 0;
      *maxModMeanRatio = 0;
      *meanModMeanRatio = 0;
      *minModCondNum = 0;
      *maxModCondNum = 0;
      *meanModCondNum = 0;
   }
   else {
      *minContent = minCont;
      *maxContent = maxCont;
      *meanContent = sumCont / numSimplices;
      *minDeterminant = minDet;
      *maxDeterminant = maxDet;
      *meanDeterminant = sumDet / numSimplices;
      *minModMeanRatio = minMmr;
      *maxModMeanRatio = maxMmr;
      *meanModMeanRatio = sumMmr / numSimplices;
      *minModCondNum = minMcn;
      *maxModCondNum = maxMcn;
      *meanModCondNum = sumMcn / numSimplices;
   }
}


// Calculate quality statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
computeQualityStatistics(const IndSimpSet<N, M, T>& iss,
                         T* minContent,
                         T* maxContent,
                         T* meanContent,
                         T* minDeterminant,
                         T* maxDeterminant,
                         T* meanDeterminant,
                         T* minModMeanRatio,
                         T* maxModMeanRatio,
                         T* meanModMeanRatio,
                         T* minModCondNum,
                         T* maxModCondNum,
                         T* meanModCondNum) {
   computeQualityStatistics<M>(iss.vertices.begin(),
                               iss.indexedSimplices.begin(),
                               iss.indexedSimplices.end(),
                               minContent,
                               maxContent,
                               meanContent,
                               minDeterminant,
                               maxDeterminant,
                               meanDeterminant,
                               minModMeanRatio,
                               maxModMeanRatio,
                               meanModMeanRatio,
                               minModCondNum,
                               maxModCondNum,
                               meanModCondNum);
}







// Print quality statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T, typename VertRAIter, typename ISForwardIter>
inline
void
printQualityStatistics(std::ostream& out,
                       VertRAIter verticesBeginning, VertRAIter verticesEnd,
                       ISForwardIter indexedSimplicesBeginning,
                       ISForwardIter indexedSimplicesEnd) {
   typedef typename std::iterator_traits<VertRAIter>::value_type Vertex;
   typedef std::array < Vertex, M + 1 > Simplex;

   T minContent, maxContent, meanContent,
     minDeterminant, maxDeterminant, meanDeterminant,
     minModMeanRatio, maxModMeanRatio, meanModMeanRatio,
     minModCondNum, maxModCondNum, meanModCondNum;

   computeQualityStatistics<M>
   (verticesBeginning, indexedSimplicesBeginning, indexedSimplicesEnd,
    &minContent, &maxContent, &meanContent,
    &minDeterminant, &maxDeterminant, &meanDeterminant,
    &minModMeanRatio, &maxModMeanRatio, &meanModMeanRatio,
    &minModCondNum, &maxModCondNum, &meanModCondNum);

   const std::size_t numSimplices = std::distance(indexedSimplicesBeginning,
                                    indexedSimplicesEnd);
   const T content = meanContent * numSimplices;

   // Count the number of simplices with a positive determinant.
   Simplex s;
   SimplexJac<M, T> simplexJacobian;
   std::size_t numSimplicesWithPositiveDeterminant = 0;
   // Loop over the simplices.
   for (ISForwardIter i = indexedSimplicesBeginning; i != indexedSimplicesEnd;
         ++i) {
      for (std::size_t j = 0; j != s.size(); ++j) {
         s[j] = verticesBeginning[(*i)[j]];
      }
      simplexJacobian.setFunction(s);
      if (simplexJacobian.getDeterminant() > 0.0) {
         ++numSimplicesWithPositiveDeterminant;
      }
   }

   // Compute a bounding box around the mesh.
   geom::BBox<T, N> boundingBox =
     geom::specificBBox<geom::BBox<T, N> >(verticesBeginning, verticesEnd);

   // Compute the edge length statistics.
   T minimumLength, maximumLength;
   computeEdgeLengthStatistics<M>(verticesBeginning,
                                  indexedSimplicesBeginning,
                                  indexedSimplicesEnd,
                                  &minimumLength, &maximumLength);

   out << "Space dimension = " << N << "\n"
       << "Simplex dimension = " << M << "\n"
       << "Bounding box = " << boundingBox << "\n"
       << "Number of vertices = "
       << int(std::distance(verticesBeginning, verticesEnd)) << "\n"
       << "Number of simplices = " << numSimplices << "\n"
       << "Number of simplices with positive volume = "
       << numSimplicesWithPositiveDeterminant << "\n"
       << "content = " << content
       << " min = " << minContent
       << " max = " << maxContent
       << " mean = " << meanContent << "\n"
       << "determinant:"
       << " min = " << minDeterminant
       << " max = " << maxDeterminant
       << " mean = " << meanDeterminant << "\n"
       << "mod mean ratio:"
       << " min = " << minModMeanRatio
       << " max = " << maxModMeanRatio
       << " mean = " << meanModMeanRatio << "\n"
       << "mod cond num:"
       << " min = " << minModCondNum
       << " max = " << maxModCondNum
       << " mean = " << meanModCondNum << "\n"
       << "edge lengths:"
       << " min = " << minimumLength
       << " max = " << maximumLength << "\n";
}






// Print quality statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T, typename SimplexForwardIterator>
inline
void
printQualityStatistics(std::ostream& out,
                       SimplexForwardIterator simplicesBeginning,
                       SimplexForwardIterator simplicesEnd) {
   T minContent, maxContent, meanContent,
     minDeterminant, maxDeterminant, meanDeterminant,
     minModMeanRatio, maxModMeanRatio, meanModMeanRatio,
     minModCondNum, maxModCondNum, meanModCondNum;

   computeQualityStatistics<M>
   (simplicesBeginning, simplicesEnd,
    &minContent, &maxContent, &meanContent,
    &minDeterminant, &maxDeterminant, &meanDeterminant,
    &minModMeanRatio, &maxModMeanRatio, &meanModMeanRatio,
    &minModCondNum, &maxModCondNum, &meanModCondNum);

   const std::size_t numSimplices =
      std::distance(simplicesBeginning, simplicesEnd);
   const T content = meanContent * numSimplices;

   // Count the number of simplices with a positive determinant.
   // Compute a bounding box around the mesh.
   SimplexJac<M, T> simplexJacobian;
   std::size_t numSimplicesWithPositiveDeterminant = 0;
   geom::BBox<T, N> boundingBox;
   if (numSimplices != 0) {
      boundingBox.lower = (*simplicesBeginning)[0];
      boundingBox.upper = (*simplicesBeginning)[0];
   }
   // Loop over the simplices.
   for (SimplexForwardIterator i = simplicesBeginning; i != simplicesEnd; ++i) {
      simplexJacobian.setFunction(*i);
      // Check the sign of the determinant.
      if (simplexJacobian.getDeterminant() > 0.0) {
         ++numSimplicesWithPositiveDeterminant;
      }
      // Add the simplex vertices to the bounding box.
      for (std::size_t m = 0; m != M + 1; ++m) {
         boundingBox += (*i)[m];
      }
   }

   // Compute the edge length statistics.
   T minimumLength, maximumLength;
   computeEdgeLengthStatistics<M>(simplicesBeginning, simplicesEnd,
                                  &minimumLength, &maximumLength);

   out << "Space dimension = " << N << "\n"
       << "Simplex dimension = " << M << "\n"
       << "Bounding box = " << boundingBox << "\n"
       // << "Number of vertices = "
       // << int(std::distance(verticesBeginning, verticesEnd)) << "\n"
       << "Number of simplices = " << numSimplices << "\n"
       << "Number of simplices with positive volume = "
       << numSimplicesWithPositiveDeterminant << "\n"
       << "content = " << content
       << " min = " << minContent
       << " max = " << maxContent
       << " mean = " << meanContent << "\n"
       << "determinant:"
       << " min = " << minDeterminant
       << " max = " << maxDeterminant
       << " mean = " << meanDeterminant << "\n"
       << "mod mean ratio:"
       << " min = " << minModMeanRatio
       << " max = " << maxModMeanRatio
       << " mean = " << meanModMeanRatio << "\n"
       << "mod cond num:"
       << " min = " << minModCondNum
       << " max = " << maxModCondNum
       << " mean = " << meanModCondNum << "\n"
       << "edge lengths:"
       << " min = " << minimumLength
       << " max = " << maximumLength << "\n";
}




// Print quality statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
printQualityStatistics(std::ostream& out,
                       const IndSimpSet<N, M, T>& iss) {
   printQualityStatistics<N, M, T>(out, iss.vertices.begin(),
                                   iss.vertices.end(),
                                   iss.indexedSimplices.begin(),
                                   iss.indexedSimplices.end());
}






// Print information about the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
printInformation(std::ostream& out, const IndSimpSet<N, M, T>& x) {
   out << "Space dimension = " << N << "\n"
       << "Simplex dimension = " << M << "\n"
       << "Bounding box = "
       << specificBBox<BBox<T, N> >(x.vertices.begin(), x.vertices.end())
       << "\n"
       << "Number of vertices = " << x.vertices.size() << "\n"
       << "Number of simplices = " << x.indexedSimplices.size() << "\n";
}


// Print information about the mesh.
template<std::size_t N, std::size_t M, typename T>
inline
void
printInformation(std::ostream& out,
                 const IndSimpSetIncAdj<N, M, T>& x) {
   // Print information that does not depend on the incidences and adjacencies.
   printInformation(out, static_cast<const IndSimpSet<N, M, T>&>(x));

   {
      std::array < std::size_t, M + 2 > counts;
      countAdjacencies(x, &counts);
      out << "Adjacency counts = " << counts << "\n";
   }
   out << "Number of components = " << countComponents(x) << "\n";
}

} // namespace geom
}
