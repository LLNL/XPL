#!/usr/bin/env bash

sed -i 's/^float ff_variable/__constant__ float ff_variable/g' $@
sed -i 's/^::float3 ff_flux/__constant__ float3 ff_flux/g' $@
sed -i 's/((float )(1.4 - 1.0))/((float )(1.4f - 1.0f))/g' $@


