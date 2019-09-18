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
      /*   std::vector<Mappoint::Ptr>  mps = tracking->_current_frame->_map_points;
            for(Mappoint::Ptr mp:mps)
            {
                double x = mp->_T_w2p[0];
                double y = mp->_T_w2p[1];
                double z = mp->_T_w2p[2];

                if(abs(x) > 100 || abs(x)<0.01)
                {
                   cout <<"x  " << x << endl;
                }
                if(abs(y) > 100 || abs(y)<0.01)
                {
                    cout <<"y  " << y << endl;
                }
                if(abs(z) > 100 || abs(z)<2)
                {
                    cout <<"z  "<< z << endl;
                }
            }
*/          frame->_isKF = true;
            localMapping.run(frame);
            //map->insertKeyFrame(tracking->_KF);
            viewer->updateLocalMap();
        }
    }

    return 0;
}