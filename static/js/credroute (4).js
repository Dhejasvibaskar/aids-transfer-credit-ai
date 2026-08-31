document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector("[data-toggle-password]");
  if (btn) {
    btn.addEventListener("click", () => {
      const p = document.querySelector("#password");
      if (!p) return;
      p.type = p.type === "password" ? "text" : "password";
      btn.textContent = p.type === "password" ? "◉" : "◎";
    });
  }
  const slider = document.querySelector("#threshold");
  const out = document.querySelector("#thresholdValue");
  if (slider && out) {
    const sync = () => out.textContent = slider.value + "%";
    slider.addEventListener("input", sync); sync();
  }
  document.querySelectorAll(".upload-zone input[type=file]").forEach(input => {
    input.addEventListener("change", () => {
      const zone = input.closest(".upload-zone");
      if (!zone || !input.files.length) return;
      const title = zone.querySelector("strong");
      const sub = zone.querySelector("small");
      if (input.multiple && input.files.length > 1) {
        title.textContent = input.files.length + " transcripts selected";
        sub.textContent = Array.from(input.files).map(f => f.name).join(" · ");
      } else {
        title.textContent = input.files[0].name;
        sub.textContent = "Ready for evaluation";
      }
    });
  });
});
