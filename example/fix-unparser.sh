#!/usr/bin/env bash

sed -i 's/^static maxpos_t maxPos = 0;/__managed__ static maxpos_t maxPos = 0;/g' $@
