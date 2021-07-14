set +xe
set -eo pipefail
# Print `Green` color font.
gprint(){
  echo -e "\033[42;37m $1 \033[0m"
}
# Print `Red` color font.
rprint(){
echo -e "\033[31m $1 \033[0m"
}