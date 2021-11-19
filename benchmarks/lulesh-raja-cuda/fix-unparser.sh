#!/usr/bin/env bash

sed -i 's/ \*  \* / \* __restrict__ \* /g' $@
sed -i 's/^inline static void CalcElemShapeFunctionDerivatives/inline static __device__ void CalcElemShapeFunctionDerivatives/g' $@
sed -i 's/^inline static void CalcElemNodeNormals/inline static __device__ void CalcElemNodeNormals/g' $@
sed -i 's/^inline static void SumElemStressesToNodeForces/inline static __device__ void SumElemStressesToNodeForces/g' $@
sed -i 's/^inline static void CalcElemFBHourglassForce/inline static __device__ void CalcElemFBHourglassForce/g' $@
sed -i 's/^inline static void CalcElemVolumeDerivative/inline static __device__ void CalcElemVolumeDerivative/g' $@
sed -i 's/^inline static void SumElemFaceNormal/inline static __device__ void SumElemFaceNormal/g' $@
sed -i 's/^inline static void VoluDer/inline static __device__ void VoluDer/g' $@
sed -i 's/^inline static Real_t AreaFace/inline static __device__ Real_t AreaFace/g' $@

sed -i 's/^inline static Real_t CalcElemCharacteristicLength/inline static __device__ Real_t CalcElemCharacteristicLength/g' $@
sed -i 's/^inline static void CalcElemShapeFunctionDerivatives/inline static __device__ void CalcElemShapeFunctionDerivatives/g' $@
sed -i 's/^inline static void CalcElemVelocityGradient/inline static __device__ void CalcElemVelocityGradient/g' $@

# CalcElemVolume
sed -i 's/Real_t CalcElemVolume/__host__ __device__ Real_t CalcElemVolume/g' $@

sed -i 's/^inline real4 SQRT/inline __host__ __device__ real4 SQRT/g' $@
sed -i 's/^inline real8 SQRT/inline __host__ __device__ real8 SQRT/g' $@
sed -i 's/^inline real10 SQRT/inline __host__ __device__ real10 SQRT/g' $@

sed -i 's/^inline real4 CBRT/inline __host__ __device__ real4 CBRT/g' $@
sed -i 's/^inline real8 CBRT/inline __host__ __device__ real8 CBRT/g' $@
sed -i 's/^inline real10 CBRT/inline __host__ __device__ real10 CBRT/g' $@

sed -i 's/^inline real4 FABS/inline __host__ __device__ real4 FABS/g' $@
sed -i 's/^inline real8 FABS/inline __host__ __device__ real8 FABS/g' $@
sed -i 's/^inline real10 FABS/inline __host__ __device__ real10 FABS/g' $@

