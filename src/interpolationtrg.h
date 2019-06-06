double interpolationtrg(int i, double x, double y)
{
    if (i==0){return 1-x-y;}
    else if (i==1){return x;}
    else {
        return y;
    }
}
