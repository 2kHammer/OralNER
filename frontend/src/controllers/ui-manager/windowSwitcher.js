function switchWindow(id) {
    // hide every window
    document.querySelectorAll('.window').forEach(div => {
        div.classList.remove('active')
    });
    // view selected window
    const active = document.getElementById(id);
    if (active) {
      active.classList.add('active');
    }
  }
  