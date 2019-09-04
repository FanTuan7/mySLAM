#ifndef LOCALBA_H
#define LOCALBA_H
#include "map.h"

class LocalBA
{
    public:
    using Ptr = std::shared_ptr<LocalBA>;
    using ConstPtr = std::shared_ptr<const LocalBA>;
    LocalBA(Map::Ptr map);
    Map::Ptr _map;
};

#endif