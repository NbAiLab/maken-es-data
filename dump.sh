# Dumps all records and objects (books and images) since 2000-01-01 by week
# $1 output-dir

if (($#<1)) then
  echo "An output-dir must be passed"
else
  OUTPUT=$(realpath ${1})
fi

INIT_DATE="20000101"
diff=($(date +%s -d "today")-$(date +%s -d ${INIT_DATE}))/$((7*24*3600))
for i in $(eval echo {$diff..1}); do
  start=$(($i*7))
  end=$(($start-6))
  START_DATE=$(date -d "last monday-$start days" +"%Y%m%d")
  END_DATE=$(date -d "last monday-$end day" +"%Y%m%d")

  echo
  echo "===================="
  echo "| $START_DATE $END_DATE |"
  echo "===================="
  echo

  WEEK_FOLDER="$OUTPUT/$START_DATE-$END_DATE"

  mkdir -p $WEEK_FOLDER

  echo
  echo "-----------"
  echo "| IMAGES  |"
  echo "-----------"
  echo

  echo "Downloading images records"
  mkdir -p $WEEK_FOLDER/images/records
  python api_downloader.py \
    -f "firstDigitalContentTime:[$START_DATE TO $END_DATE]" \
    -f mediatype:bilder \
    -f contentClasses:JP2 \
    -f digital:Ja \
    --groupby "accessInfo.accessAllowedFrom,metadata.originInfo.firstDigitalContentTime:DATE" \
    --size 100 \
    --scroll \
    --delay 0.0 \
    --log \
    -o $WEEK_FOLDER/images/records

  echo "Download images"
  mkdir -p $WEEK_FOLDER/images/vectors
  python image_vectorizer.py \
    $WEEK_FOLDER/images/records "**/**/*" \
    $WEEK_FOLDER/images/vectors \
    --download
    --no-inference
    --search_local_paths "/nfsmounts/meta2/disk4/folder1/nancy/maken/catalog/images-vectors,/nfsmounts/meta2/disk4/folder1/nancy/maken/vectors_20211011/images"

  echo
  echo "----------"
  echo "| BOOKS  |"
  echo "----------"
  echo

  echo "Downloading books records"
  python api_downloader.py \
    -f "firstDigitalContentTime:[$START_DATE TO $END_DATE]" \
    -f mediatype:bøker \
    --groupby "accessInfo.accessAllowedFrom,metadata.originInfo.firstDigitalContentTime:DATE" \
    --size 100 \
    --scroll \
    --delay 0.0 \
    --log \
    -o $WEEK_FOLDER/records/books

  echo "Downloading books"
  mkdir -p $WEEK_FOLDER/books/vectors
  python image_vectorizer.py \
    $WEEK_FOLDER/books/records "**/**/*" \
    $WEEK_FOLDER/books/vectors \
    --download
    --no-inference
    --search_local_paths "/nfsmounts/meta2/disk1/content/boktekst/"

done
