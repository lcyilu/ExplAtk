#! /bin/bash

CURPATH=$( pwd )
# echo $CURPATH

PROJECTPATH=$CURPATH/ 
TXLCODEPATH="../Txl/"  
COUNTRESULTPATH="../CountResult/"  

function getAvailableActions() {
    local ACTIONS_FILE=$COUNTRESULTPATH"trans_actions.txt"
    # 读取每一行非空且为数字的变换编号
    grep -E '^[0-9]+$' $ACTIONS_FILE | sort -n | uniq
}


 function muteCode(){ 
    TRANSFORMCODE=$1
    ACTION=$2 
    # cd $PROJECTPATH  

    # 进行指定类型变换
    txl   -q -s 128  $TRANSFORMCODE $TXLCODEPATH"RemoveCompoundStateSemicolon.Txl" > temp0.c  &&
    txl   -q -s 128 temp0.c $TXLCODEPATH"RemoveNullStatements.Txl" > temp00.c &&
    case ${ACTION} in 
        1)
            # echo "txl   -q -s 128  temp00.c $TXLCODEPATH"1ChangeRename.Txl" > temp1.c"
            txl  -q -s 128  temp00.c $TXLCODEPATH"1ChangeRename.Txl" > temp1.c  
            ;;
        2)
            txl   -q -s 128  temp00.c $TXLCODEPATH"2A3ChangeCompoundForAndWhile.Txl" > temp1.c
            ;;
        3)
            txl   -q -s 128  temp00.c $TXLCODEPATH"2A3ChangeCompoundForAndWhile.Txl" > temp1.c
            ;;
        4)
            txl   -q -s 128  temp00.c $TXLCODEPATH"4changeCompoundDoWhile.Txl" > temp1.c
            ;;
        5)
            txl   -q -s 128  temp00.c $TXLCODEPATH"5A6changeCompoundIf.Txl" > temp1.c
            ;;
        6)
            txl   -q -s 128  temp00.c $TXLCODEPATH"5A6changeCompoundIf.Txl" > temp1.c 
            ;;
        7)
            txl   -q -s 128  temp00.c $TXLCODEPATH"7changeCompoundSwitch.Txl" > temp1.c
            ;;
        8)
            txl   -q -s 128  temp00.c $TXLCODEPATH"8changeCompoundLogicalOperator.Txl" > temp1.c 
            ;;
        9)  
            txl   -q -s 128  temp00.c $TXLCODEPATH"9changeSelfOperator.Txl" > temp1.c 
            ;;
        10)
            txl   -q -s 128  temp00.c $TXLCODEPATH"10changeCompoundIncrement.Txl" > temp1.c 
            ;;
        11)
            txl   -q -s 128  temp00.c $TXLCODEPATH"11changeConstant.Txl" > temp1.c 
            ;;
        12)
            txl   -q -s 128  temp00.c $TXLCODEPATH"12changeVariableDefinitions.Txl" > temp1.c 
            ;;
        13)
            txl   -q -s 128  temp00.c $TXLCODEPATH"13changeAddJunkCode.Txl" > temp1.c 
            ;;
        14)
            txl   -q -s 128  temp00.c $TXLCODEPATH"14changeExchangeCodeOrder.Txl" > temp1.c 
            ;;
        15)
            txl   -q -s 128  temp00.c $TXLCODEPATH"15changeDeleteCode.Txl" > temp1.c 
            ;;
        *)
        exit 1 
        ;;  
    esac  
    txl   -q -s 128 temp1.c $TXLCODEPATH"RemoveNullStatements.Txl" > temp3.c &&
    txl   -q -s 128 temp3.c $TXLCODEPATH"PrettyPrint.Txl" > temp4.c &&
    txl   -q -s 128 temp4.c $TXLCODEPATH"RemoveNullStatements.Txl" > temp.c &&  
    # python ParseCode.py temp.c &&
    echo "result reserved in $3"
    cp temp.c "$3"
    rm  -rf temp*
    
}

# 批量处理函数
function batchProcess(){
    filelist=$1
    out_dir=$2
    while IFS= read -r code_path; do
        # 跳过空行
        [[ -z "$code_path" ]] && continue
        # 调用原有 main 逻辑
        main "$code_path" "$out_dir"
    done < "$filelist"
}

function main(){
    code_path=$1
    filename=$(basename "$code_path")
    extension="${filename##*.}"
    name="${filename%.*}" 
    out_dir=$2

    txl   -q $code_path $TXLCODEPATH"CountModification.Txl" > /dev/null 2> /dev/null &&
    python GenRandomChange.py $COUNTRESULTPATH

    actions=$(getAvailableActions)

        
    for action in $actions; do
        outname=$out_dir"${name}_${action}.${extension}"
        muteCode "$code_path" "$action" "$outname"
    done    
}

# 如果参数为2个，单文件处理；如果为3个，批量处理
if [[ $# -eq 2 ]]; then
    main "$1" "$2"
elif [[ $# -eq 3 ]]; then
    batchProcess "$1" "$2"
else
    echo "用法: $0 <源文件路径> <输出目录> 或 $0 <文件列表txt> <输出目录> <batch>"
    exit 1
fi
 



 
 
