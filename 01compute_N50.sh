contig_file=$1
awk '/^>/ {if (seqlen) print seqlen; seqlen=0; next} {seqlen+=length($0)} END {print seqlen}' "$contig_file" | sort -rn > ${contig_file}_lengths.txt

# CAL N50
total_length=$(awk '{sum+=$1} END {print sum}' ${contig_file}_lengths.txt)
target_length=$(awk -v total="$total_length" 'BEGIN {cutoff=total/2; current=0} {current+=$1; if (current >= cutoff) {print $1; exit}}' ${contig_file}_lengths.txt)

# N50
echo "N50: $target_length"
