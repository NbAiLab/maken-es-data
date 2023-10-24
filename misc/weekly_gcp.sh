echo "Cron job started"

START_DATE_LAST_WEEK=$(date -d "last monday-7 days" +"%Y%m%d")
END_DATE_LAST_WEEK=$(date -d "last monday-1 day" +"%Y%m%d")

START_DATE="${1:-$START_DATE_LAST_WEEK}"
END_DATE="${2:-$END_DATE_LAST_WEEK}"
BUCKET="${3:-gs://maken-vectors}"

echo "Time period: $START_DATE to $END_DATE"

WEEK_FOLDER="$START_DATE-$END_DATE"

mkdir -p $WEEK_FOLDER

echo
echo "-----------"
echo "| IMAGES  |"
echo "-----------"
echo

echo "Downloading images records"
mkdir -p $WEEK_FOLDER/images/records
python downloader.py \
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

echo "Creating image vectors"
mkdir -p $WEEK_FOLDER/images/vectors
python image_vectorizer.py \
  $WEEK_FOLDER/images/records "**/**/*" \
  $WEEK_FOLDER/images/vectors \
  --objects_dir $WEEK_FOLDER/images/objects \
  --vector_format npy \
  --n_jobs 32 \
  --batch 1000

echo "Uploading images records and vectors"
gsutil -m rsync -am --include='*.npy' --include='*.json' --include='*/' --exclude='*' $WEEK_FOLDER/images $BUCKET/$WEEK_FOLDER/images


echo
echo "----------"
echo "| BOOKS  |"
echo "----------"
echo

echo "Downloading books records"
python downloader.py \
  -f "firstDigitalContentTime:[$START_DATE TO $END_DATE]" \
  -f mediatype:bøker \
  --groupby "accessInfo.accessAllowedFrom,metadata.originInfo.firstDigitalContentTime:DATE" \
  --size 100 \
  --scroll \
  --delay 0.0 \
  --log \
  -o $WEEK_FOLDER/records/books

echo "Creating books vectors"
python book_vectorizer.py \
  $WEEK_FOLDER/records/books "**/**/*" \
  $WEEK_FOLDER/vectors/books \
  --objects_dir $WEEK_FOLDER/images/objects \
  --vector_format json
#  npy _inceptionv3 false true 100 1000 0

echo "Uploading books vectors"
gsutil -m cp -r $WEEK_FOLDER/vectors/books $BUCKET/$WEEK_FOLDER/
