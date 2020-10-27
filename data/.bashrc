# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi


alias mosek="~/Documents/Mosek/mosek/7/tools/platform/linux64x86/bin/mosek"


# Lazy git function to perform add, commit and push in one line
function lazygit() {
    git add .
    git commit -a -m "$1"
    git push
}

# Copy Lab notebook to previous folder + lazygit 
function updateLabNotebook() {
    # Store original directory
    mypwd=$(pwd) 

    # Update Lab Notebook pdfs
    cd /home/shurjo/Documents/Lab/labnb_shurjo 
    for d in */ ; do
        cd "$d"
        for i in */; do
            if [ "$i" == "Makefiles/" ]; then
                cp Makefiles/*.pdf .
            fi
        done
        cd ..    
    done

    # Update the project
    git add .
    git commit -a -m "$1"
    git push

    # cd back to the original directory
    cd $mypwd

}

# Python doesn't link to its site packages automatically for some reason
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/dist-packages/

# Set Eigen's environmental variable

# Set Apriltags Environmental variable
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:/home/shurjo/Documents/TestCode/cpp-scripts/apriltags/build/lib/pkgconfig

# Bash RC Aliases
alias BASHRC="vim ~/.bashrc"
alias VIMRC="vim ~/.vimrc"
alias DEV="cd ~/Documents/dev"
alias SOURCE="source ~/.bashrc"
alias TMUX="vim ~/.tmux.conf"

# Copy and paste using xclip
alias "c=xclip"
alias "v=xclip -o"

# Common word directories
alias WAPP="cd /home/shurjo/Documents/dev/ARR-System/Webapp-Module"
alias VAPP="cd /home/shurjo/Documents/dev/ARR-System/Data-Storage-Module/build"
alias TEST="cd /home/shurjo/Documents/dev/test-scripts"
alias WRITE="cd /home/shurjo/Documents/dev/ARR-System/Documentation/Makefiles"
alias ZSLANG="cd /z/home/shurjo/robotslang"
alias ZSLANGV="cd /z/home/shurjo/robotslang_video/"
alias NAU="nautilus ."
alias GO="gnome-open"
alias TA="tmux attach"
#alias MOD="module load cuda cudnn gflags tensorflow"
alias RLTEXT="cd /z/home/shurjo/rl-texture-simulations"
# Rosdep stuff
#source /opt/ros/indigo/setup.bash
#source ~/rosbuild_ws/setup.bash
#source /home/shurjo/catkin_ws/devel/setup.bash

# PBS Script stuff


alias ZSHURJO="cd /z/home/shurjo"

# Bazel
alias bazel="JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 bazel"

# Cluster (machine of last jobs)
alias WMAC="qstat -f -u shurjo | grep exec"
alias CMAC="qstat -f -u shurjo | grep exec_host"
alias WMACS="~/.wmac.sh"
alias NGPUS="qselect -s R | xargs qstat -f | grep exec_gpus | wc -l"
alias CSTATUS="pbsnodes -l all"
alias IMPL="cd /z/home/shurjo/implicit-mapping/openai-a3c-impl"
IMPL="/z/home/shurjo/implicit-mapping/openai-a3c-impl"
alias MAPS="cd /z/home/shurjo/implicit-mapping/deepmind-lab/assets/game_scripts"
EXP="/z/home/shurjo/implicit-mapping/a3c-random-mazes"
ZEXP="/z/home/shurjo/implicit-mapping/a3c-random-mazes"
alias EXP="cd /z/home/shurjo/implicit-mapping/a3c-random-mazes"
alias EXPL="cd $EXP"

function latest() {
    t=$(ls -t $EXP | head -1)
    cd "$EXP/$t"
}
function zlatest() {
    t=$(ls -t $ZEXP | head -1)
    cd "$ZEXP/$t"
}
function logtail() {
    t=$(ls -t $EXP | head -1)
    fl="$EXP/$t/logs/train/a3c.w-0.out"
   	while [ ! -f $fl ]; do 
		sleep 1
	done 
    tail -f $fl
}
function dlogtail() {
    t=$(ls -t $EXP | head -1)
    fl="$EXP/$t/logs/dataset/a3c.w-0.out"
   	while [ ! -f $fl ]; do 
		sleep 1
	done 
    tail -f $fl
}
function tlogtail() {
    t=$(ls -t $EXP | head -1)
    fl="$EXP/$t/logs/test/a3c.w-0.out"
   	while [ ! -f $fl ]; do 
		sleep 1
	done 
    tail -f $fl
}

function imgtail() {
    t=$(ls -t /tmp/shurjo-maps | head -1)
    nautilus /tmp/shurjo-maps/$t/videos/test/worker-00/00000000 
}

alias WEB="cd /z/home/shurjo/deep-rl-visualization"
alias WRITE="cd /z/home/shurjo/writing"
alias LOG="latest; vim logs/train/a3c.w-0.out"
alias TLOG="latest; vim logs/test/a3c.w-0.out"
alias LOGTAIL="logtail" #"latest; tail -f logs/train/a3c.w-0.out"
alias TLOGTAIL="tlogtail" #"latest; tail -f logs/test/a3c.w-0.out"


# Python debugger
alias pythonb="python -mpdb -c \"c\""

# PF
alias PF="cd /z/home/shurjo/writing/BaDhLOCLANG-RSS2018/code; source venv/bin/activate; PYTHONPATH=""; cd ~/projects/rslang-localization/"
alias VENV="source venv/bin/activate; PYTHONPATH="""

# Cluster stuff
function load_z(){
    if [ -f /z/sw/Modules/default/init/bash ]; then
        source /etc/profile.d/modules.sh
        #module load cuda cudnn gflags numpy scipy 
        export PATH=/z/sw/bin/:/z/sw/sbin/:$PATH
        export LD_LIBRARY_PATH=/z/sw/lib/:$LD_LIBRARY_PATH
        export MANPATH=/z/sw/share/man/:$MANPATH
        # tensorflow 1.0
        #PYTHONPATH=/z/home/shurjo/sw/lib/python2.7/site-packages/:$PYTHONPATH
        #export PATH="/z/home/shurjo/sw/bazel/bin:$PATH"
    fi
}

#alias grepR=

function grepR(){
    grep -R --exclude-dir=env --exclude-dir=logs "$1" .
}

timeout 1 bash -c 'ls /z/ > /dev/null'
exit_status=$?

if [[ $exit_status -eq 124 ]]; then
   echo "Error: Unable to connect to Z"
#else
    #echo "Not loading Z"
	#load_z
fi


# virtualenv and virtualenvwrapper
#export WORKON_HOME=/home/shurjo/.virtualenvs
#source /usr/local/bin/virtualenvwrapper.sh
#export GOPATH=${HOME}/go
#export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin

#. /usr/local/etc/bash_completion.d/singularity

# MUJOCO
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shurjo/.mujoco/mjpro150/bin

function findR(){
    if [ "$#" -ne 1 ]; then
        find . -name "$1"  -not -path "./env/*"
    else
        find $1 -name "$2"  -not -path "./env/*"

    fi
}


function for5(){
    for seed in $(seq 1 5); do
        echo "$@ --seed $seed"
        "$@" --seed $seed
    done
}

alias sizes='du -hs * | sort -'
alias SQ='squeue'
alias qstat='squeue'
