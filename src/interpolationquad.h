double interpolationquad(int i, double x, double y)
{
    if (i==0){return 1.0/4*(1+x)*(1+y);}
    else if (i==1){return 1.0/4*(1-x)*(1+y);}
    else if (i==2){return 1.0/4*(1-x)*(1-y);}
    else{return 1.0/4*(1+x)*(1-y);}
}
