#include "frame.h"
#include "dataset.h"
#include "tracking.h"
#include "map.h"
//#include "local_mapping_custom_type.h"
#include "local_mapping_g2o_type.h"
#include "viewer.h"
#include <iostream>

int main()
{   

    Dataset dataset = Dataset("../example/config.yaml");

    cout << "start loading vocab" << endl;
    dataset.loadDBoW3();
    cout << "finish loading vocab" << endl;

    Map::Ptr map(new Map());

    Viewer::Ptr viewer(new Viewer());
    viewer->setMap(map);
    viewer->_groundtruth = dataset._groundtruth;

    LocalMapping_g2o localMapping(map);
    
    Tracking::Ptr tracking(new Tracking(dataset._camera,
                                        map,
                                        dataset._features,
                                        dataset._scaleFactor,
                                        dataset._levels,
                                        dataset._iniThFAST,
                                        dataset._minThFAST,
                                        dataset._KF_DoWrate_Low,
                                        dataset._KF_DoWrate_High,
                                        dataset._KF_mindistance,
                                        dataset._KF_maxdistance,
                                        dataset._vocab));

    while (1)
    {
        Frame::Ptr frame = dataset.NextFrame();
        if (frame!=nullptr )
        {  
          tracking->addFrame(frame);

          //  localMapping.run(frame);
          if(frame->_isKF)
          {
               map->insertKeyFrame(frame);
               for(auto mp:frame->_map_points)
               {
                   map->insertMapPoint(mp);
               }
          }
           
            viewer->updateLocalMap();
        }
    }

    return 0;
}