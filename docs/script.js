(() => {
  const progress = document.querySelector('#progress-bar');
  const updateProgress = () => {
    const range = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${range > 0 ? (window.scrollY / range) * 100 : 0}%`;
  };
  updateProgress();
  window.addEventListener('scroll', updateProgress, { passive: true });
  window.addEventListener('resize', updateProgress);

  const navToggle = document.querySelector('#nav-toggle');
  const nav = document.querySelector('#nav-links');
  navToggle.addEventListener('click', () => {
    const open = navToggle.getAttribute('aria-expanded') !== 'true';
    navToggle.setAttribute('aria-expanded', String(open));
    navToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    nav.classList.toggle('is-open', open);
  });
  nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
    nav.classList.remove('is-open');
    navToggle.setAttribute('aria-expanded', 'false');
    navToggle.setAttribute('aria-label', 'Open navigation');
  }));

  const revealItems = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -35px 0px' });
    revealItems.forEach(item => observer.observe(item));
  } else {
    revealItems.forEach(item => item.classList.add('is-visible'));
  }

  const dialog = document.querySelector('#figure-dialog');
  const dialogImage = document.querySelector('#figure-dialog-image');
  const dialogTitle = document.querySelector('#figure-dialog-title');
  document.querySelectorAll('.gallery-trigger').forEach(button => {
    button.addEventListener('click', () => {
      dialogImage.src = button.dataset.figureSrc;
      dialogImage.alt = button.dataset.figureAlt;
      dialogTitle.textContent = `DASA / ${button.dataset.figureTitle}`;
      dialog.showModal();
    });
  });
  document.querySelector('#figure-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target === dialog) dialog.close();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && dialog.open) dialog.close();
  });

})();
