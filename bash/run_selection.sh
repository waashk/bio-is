cd $WORKDIR

datain="$WORKDIR/resources/datasets"
out="$WORKDIR/resources/outsel"

mkdir -p $out

datasets=(aisopos_ntua_2L)
methods=(bio-is)


for d in ${datasets[@]};
do
    echo $d ; 
    for method in ${methods[@]} 
    do
        echo $method ;
        python3 run\_generateSplit.py -d $d -m $method --datain $datain --out $out;
    done;
done;