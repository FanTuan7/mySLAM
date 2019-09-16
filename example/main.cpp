#include "frame.h"
#include "dataset.h"
#include "tracking.h"
#include "map.h"
#include "local_mapping.h"
#include "viewer.h"

//#include <yaml.h> //为了读取config.yaml文件

#include <iostream>

int main()
{
    Dataset dataset = Dataset("../example/config.yaml");

    cout << "start loading vocab" << endl;
    dataset.loadDBoW3();
    cout << "finish loading vocab" << endl;

    Map::Ptr map(new Map());

    //LocalMapping localMapping(map);

    Viewer::Ptr viewer(new Viewer());
    viewer->setMap(map);

    Tracking::Ptr tracking(new Tracking(dataset._camera,
                                        map,
                                        dataset._features,
                                        dataset._scaleFactor,
                                        dataset._levels,
                                        dataset._iniThFAST,
                                        dataset._minThFAST,
                                        dataset._KF_DoWrate,
                                        dataset._KF_mindistance,
                                        dataset._vocab));

    while (1)
    {
        Frame::Ptr frame = dataset.NextFrame();
        tracking->addFrame(frame);
        //localMapping.readKF(tracking->getKF());
        //localMapping.insertKF();
        //localMapping.run();
        viewer->updateLocalMap();
    }

    return 0;
}