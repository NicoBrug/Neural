#pragma once
#ifndef PLOTTER_H 
#define PLOTTER_H

#include "../activation.h"
#include <sciplot/sciplot.hpp>


namespace Neural
{
    class Plotter {
        public:
            Plotter();
            void Plot(Activation *a);
        protected:

    };

}
#endif