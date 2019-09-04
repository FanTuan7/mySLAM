#include "frame.h"
#include "dataset.h"
#include "tracking.h"
#include "map.h"
//#include "viewer.h"
#include <iostream>
#include "viewer.h"

int main()
{   
   Dataset dataset("../data/00/");
/*
   cout << "start loading vocab" << endl; 
   dataset.loadDBoW3("../Vocabulary/ORBvoc.txt");
   cout << "finish loading vocab" << endl; 
*/

   Map::Ptr map(new Map());
   
   Viewer::Ptr viewer(new Viewer());
   viewer->setMap(map);

   Tracking::Ptr tracking(new Tracking(map));
   tracking->_viewer = viewer;

    while(1)
    {    
        Frame::Ptr frame = dataset.NextFrame();
        tracking->addFrame(frame);
    }

    return 0;
}