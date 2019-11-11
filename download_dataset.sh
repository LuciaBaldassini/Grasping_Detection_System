#!/bin/bash
echo "Downloading dataset from http://pr.cs.cornell.edu/grasping/rect_data/data.php"

curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data01.tar.gz -o data01.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data02.tar.gz -o data02.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data03.tar.gz -o data03.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data04.tar.gz -o data04.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data05.tar.gz -o data05.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data06.tar.gz -o data06.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data07.tar.gz -o data07.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data08.tar.gz -o data08.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data09.tar.gz -o data09.tar.gz
curl http://pr.cs.cornell.edu/grasping/rect_data/temp/data10.tar.gz -o data10.tar.gz

for f in *.tar.gz; do
  echo "Extracting file $f"
  tar -xzf "$f"
done

echo "Moving everything to /dataset"
mv 01/* 02/* 03/* 04/* 05/* 06/* 07/* 08/* 09/* 10/* dataset/
rm -rf 0* 10/
rm data0* data10.tar.gz
