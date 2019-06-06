void derinterpquad(int i, double x, double y, double grad[2])
{
    if (i==0)
    {
        grad[0]=(1+y)/4;
        grad[1]=(1+x)/4;
    }
    else if (i==1)
    {
        grad[0]=-(1+y)/4;
        grad[1]=(1-x)/4;
    }
    else if (i==2)
    {
        grad[0]=-(1-y)/4;
        grad[1]=-(1-x)/4;
    }
    else {
        grad[0]=(1-y)/4;
        grad[1]=-(1+x)/4;
    }
}
