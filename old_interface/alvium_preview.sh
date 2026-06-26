#!/bin/bash

# input parameters
DIR="/home/polocalc/data/alvium_test/focus" # default focus directory
PATTERN="frame_*.png"
REFRESH=0.9

# camera settings
EXPOSURE=20000
MODE="trigger"
FRAMERATE=1
ROI="1/16"

# start alvium acquisition
alvium --processing --output $DIR --mode $MODE --framerate $FRAMERATE --roi $ROI --exposure $EXPOSURE & ACQ_PID=$!
sleep 10

# cleanup on exit
cleanup() {
	echo "Stopping..."
	kill $ACQ_PID 2>/dev/null
	kill $FEH_PID 2>/dev/null
	echo "Deleting files..."
	rm -r $DIR
	exit
}
trap cleanup SIGINT SIGTERM

# this function checks if the last frame is still incomplete
is_complete() {
	file1="$1"
	size1=$(stat -c%s "$file1" 2>/dev/null)
	sleep 0.1
	size2=$(stat -c%s "$file1" 2>/dev/null)
	[ "$size1" -eq "$size2" ]
}


# create a latest.png file before calling feh
latest=$(ls -t "$DIR"/$PATTERN 2>/dev/null | head -n 1)
ln -sf "$latest" "$DIR/latest.png"

# start feh
feh --reload $REFRESH --scale-down "$DIR/latest.png" & FEH_PID=$!

# update loop
prev=""

while true; do
	latest=""
	for f in $(ls -t "$DIR"/frame_*.png); do
		if is_complete "$f"; then
			latest="$f"
			break
		fi
	done
	
	if [ -n "$latest" ] && [ "$latest" != "$prev" ]; then
		ln -sf "$latest" "$DIR/latest.png"
		prev="$latest"
	fi

	sleep $REFRESH
done
