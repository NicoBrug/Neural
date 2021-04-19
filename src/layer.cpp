
#include "../includes/layer.h"

/** Checks if the current layer has weights
 * 
 *  @return A bool : true if AsWeights, False if not
 * 
 */
bool Layer::AsWeights(){
    return this->m_as_weight;
}