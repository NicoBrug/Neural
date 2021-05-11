/**
 * \file layer.cpp
 * \brief   Base layer
 * \author Brugie Nicolas
 * \version 0.1
 *
 * Base class for all the layers
 *
 */
#include "../../includes/layers/layer.h"
using namespace Neural;

/** Checks if the current layer has weights
 * 
 *  @return A bool : true if AsWeights, False if not
 * 
 */
bool Layer::AsWeights(){
    return this->m_as_weight;
}