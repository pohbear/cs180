// Keep the outer button fixed so rotating/shrinking its image never loses hover.
// Native buttons also supply Enter/Space activation and touch support.
document.querySelectorAll('.illusion-toggle').forEach(button => {
  button.addEventListener('click', () => {
    const active = button.classList.toggle('is-active');
    button.setAttribute('aria-pressed', String(active));
  });
});
