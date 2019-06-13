void derinterpquad(int i, double x, double y, std::vector<double> &grad)
{
    //calculates the gradient of the function associated to the summit i, at the point of coordinates x,y and returns it in grad

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
