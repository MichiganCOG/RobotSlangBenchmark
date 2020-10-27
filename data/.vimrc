
"""""""""""""""""""""""""""""""""""
" 4 Spaces per indenct and all that
"""""""""""""""""""""""""""""""""""
filetype plugin indent on
" show existing tab with 4 spaces width
set tabstop=4
" when indenting with '>', use 4 spaces width
set shiftwidth=4
" On pressing tab, insert 4 spaces
set expandtab

"""""""""""""""""""""""""""""""""""
" Set Line Numbers
"""""""""""""""""""""""""""""""""""
set number

"""""""""""""""""""""""""""""""""""
" Set Mouse Control 
"""""""""""""""""""""""""""""""""""
set mouse=a
set incsearch

"""""""""""""""""""""""""""""""""""
" Pathogen 
"""""""""""""""""""""""""""""""""""
"execute pathogen#infect()
"call pathogen#helptags() " generate helptags for everything in 'runtimepath'
"syntax on
"filetype plugin indent on

" Set Solarized Dark Theme 
"""""""""""""""""""""""""""""""""""
"syntax enable
"set background=dark
"colorscheme solarized
" LCM
au BufRead,BufNewFile *.lcm set filetype=cpp


" Clear the latex italics styling
hi clear texItalStyle

" Tex width = 72
autocmd FileType tex    set textwidth=72

set clipboard=unnamedplus

" Use alt to move between splits
nmap <silent> <A-Up> :wincmd k<CR>
nmap <silent> <A-Down> :wincmd j<CR>
nmap <silent> <A-Left> :wincmd h<CR>
nmap <silent> <A-Right> :wincmd l<CR>

" PDB commands (game changer)
noremap <leader>p oimport ipdb; ipdb.set_trace()<Esc>
noremap <leader>P Oimport ipdb; ipdb.set_trace()<Esc>

"noremap <leader>p o<Esc>o<OBJ><Esc>2o<Esc>
"noremap <leader>t bi<TAR> <Esc>

au BufNewFile,BufRead *.cu set ft=cu
