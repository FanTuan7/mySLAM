#include "frame.h"

/*Frame::Frame()
{
}
*/
Frame::Frame(long id, cv::Mat left, cv::Mat right)
    : _id(id), _img_left(left), _img_right(right)
{   
    _min_x = 0;
    _max_x = _img_left.cols;
    _min_y = 0;
    _max_y = _img_left.rows;

    _gird_width_inv=  FRAME_GRID_COLS/(_max_x-_min_x);
    _grid_height_inv = FRAME_GRID_ROWS/(_max_y-_min_y);


}

void Frame::AssignKeypointsToGrid()
{
    unsigned int size = _kps_left.size();
    unsigned int reserve = size/(FRAME_GRID_ROWS*FRAME_GRID_COLS);
    
    for(int i=0; i<FRAME_GRID_COLS;i++)
    {
        for (int j=0; j<FRAME_GRID_ROWS;j++)
            _Grid[i][j].reserve(reserve);
    }

    for(int i=0;i<size;i++)
    {
        const cv::KeyPoint &kp = _kps_left[i];

        int x, y;
        if(PosInGrid(kp,x,y))
        {
            _Grid[x][y].push_back(i);
        }        
    }
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round(kp.pt.x*_gird_width_inv);
    posY = round(kp.pt.y*_grid_height_inv);

    //双目都是矫正过的 应该不可能越界
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

vector<int> Frame::getFeaturesInArea(float &x, float  &y, float &r, int minLevel, int maxLevel)
{
    vector<int> indexs;
    indexs.reserve(_kps_left.size());

    //计算区域所在的cell
    int minCellx = max( 0, (int)floor((x-r)*_gird_width_inv));
    if(minCellx>=FRAME_GRID_COLS) return indexs;

    int maxCellx = min( FRAME_GRID_COLS-1, (int)floor((x-_min_x+r)*_gird_width_inv));
    if(maxCellx<0) return indexs;

    int minCelly = max( 0, (int)floor((y-_min_y-r)*_grid_height_inv));
    if(minCelly>=FRAME_GRID_ROWS) return indexs;

    int maxCelly = min( FRAME_GRID_ROWS-1, (int)floor((y-_min_y+r)*_grid_height_inv));
    if(maxCelly<0) return indexs;

    //设定查找的金字塔层
    bool checkLevels = (minLevel>0) || (maxLevel>=0);

    //遍历所有符合范围的Cell,提取特征点的编号
    for(int u= minCellx; u<=maxCellx; u++)
    {
        for(int v = minCelly; v<=maxCelly; v++)
        {
            vector<unsigned int> cell = _Grid[u][v];
            if(cell.empty())
                continue;

            int size = cell.size();
            for(int i=0; i< size; i++)
            {
                KeyPoint &kp = _kps_left[cell[i]];
                if(checkLevels)
                {
                    /*if(kp.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kp.octave>maxLevel)
                            continue;*/
                }

                const float distx = kp.pt.x-x;
                const float disty = kp.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    indexs.push_back(cell[i]);
            }
        }
    }

    return indexs;

}   

void Frame::releaseImages()
{
    _img_left.release();
    _img_right.release();
}


