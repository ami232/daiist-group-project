function mountTrip() {
  const card = element.querySelector(".trip-card");
  if (!card) return;
  const get = selector => card.querySelector(selector);
  const data = props.value;
  const pins = get(".trip-pins");
  const canvas = get(".trip-route");
  const ctx = canvas.getContext("2d");
  const guide = get(".trip-guide");
  const checklist = get(".trip-checklist");
  const meter = get(".trip-meter");
  const details = get(".trip-details");
  const nextButton = get("[data-next]");
  const reset = get("[data-reset]");
  pins.replaceChildren();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  details.hidden = true;
  guide.hidden = checklist.hidden = !data;
  get(".trip-empty").hidden = Boolean(data);
  get("[data-example]").hidden = !data?.example;
  get("[data-map-note]").hidden = !data;
  get(".trip-stop-list").replaceChildren();
  if (!data) return;

  const picked = new Set();
  let location = "entrance";
  let active = null;
  const buttons = new Map();
  const all = data.stops.flatMap(stop => stop.items);
  meter.max = all.length;
  const unfinished = stop => stop.items.some(item => !picked.has(item.id));

  function nextStop() {
    const index = data.stops.findIndex(stop => stop.id === location);
    if (index >= 0 && unfinished(data.stops[index])) return data.stops[index];
    const ordered = index < 0 ? data.stops : [...data.stops.slice(index + 1), ...data.stops.slice(0, index)];
    return ordered.find(unfinished);
  }

  function drawLine(points, color, width) {
    if (!points || points.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(...points[0]);
    for (const point of points.slice(1)) ctx.lineTo(...point);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function redraw() {
    const next = nextStop();
    const destination = next?.id ?? "checkout";
    get("[data-to]").textContent = next?.title ?? "Checkout";
    get("[data-instruction]").textContent = destination === location ? "Pick up here" : "Walk to";
    get("[data-progress]").textContent = picked.size === all.length
      ? "All items picked up. Head to Checkout."
      : picked.size + " of " + all.length + " items picked up";
    meter.value = picked.size;
    reset.hidden = picked.size === 0;
    nextButton.hidden = !next || next.id === active;
    nextButton.textContent = next ? "Open next stop" : "";
    for (const [i, stop] of data.stops.entries()) {
      const {pin, listButton} = buttons.get(stop.id);
      const left = stop.items.filter(item => !picked.has(item.id)).length;
      pin.dataset.done = String(left === 0);
      pin.dataset.next = String(stop.id === destination && destination !== location);
      pin.setAttribute("aria-pressed", String(stop.id === active));
      listButton.setAttribute("aria-pressed", String(stop.id === active));
      listButton.dataset.done = String(left === 0);
      listButton.querySelector("small").textContent = left === 0 ? "Done" : left + " left";
      pin.setAttribute("aria-label", "Stop " + (i + 1) + ", " + stop.title + ", " + (left === 0 ? "complete" : left + " to pick up"));
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(2, 2);
    ctx.lineCap = ctx.lineJoin = "round";
    drawLine(data.trace, "#42b8b2", 5);
    const leg = data.legs[location]?.[destination];
    ctx.shadowColor = "#70f2dd";
    ctx.shadowBlur = 14;
    drawLine(leg, "#70f2dd", 6);
    ctx.shadowBlur = 0;
    const endpoint = data.points.checkout;
    ctx.beginPath();
    ctx.arc(...endpoint, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffa298";
    ctx.fill();
    if (leg) for (let i = 1; i < leg.length; i++) {
      const [x1, y1] = leg[i - 1];
      const [x2, y2] = leg[i];
      const length = Math.hypot(x2 - x1, y2 - y1);
      if (length < 38) continue;
      const ux = (x2 - x1) / length, uy = (y2 - y1) / length;
      const x = (x1 + x2) / 2, y = (y1 + y2) / 2;
      drawLine([[x - 4 * ux - 4 * uy, y - 4 * uy + 4 * ux],
                [x + 3 * ux, y + 3 * uy],
                [x - 4 * ux + 4 * uy, y - 4 * uy - 4 * ux]], "#fff", 2);
    }
    ctx.restore();
  }

  function openStop(stop, reveal) {
    active = stop.id;
    details.hidden = false;
    get("[data-stop-title]").textContent = stop.title;
    const itemList = get("[data-items]");
    itemList.replaceChildren();
    for (const item of stop.items) {
      const row = document.createElement("label");
      const checkbox = document.createElement("input");
      const name = document.createElement("span");
      name.textContent = item.name;
      checkbox.type = "checkbox";
      checkbox.checked = picked.has(item.id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
          picked.add(item.id);
          location = stop.id;
        } else picked.delete(item.id);
        redraw();
      });
      row.append(checkbox, name);
      itemList.append(row);
    }
    redraw();
    if (reveal && matchMedia("(max-width: 740px)").matches) {
      details.scrollIntoView({block: "nearest"});
    }
  }

  for (const [i, stop] of data.stops.entries()) {
    const [x, y] = data.points[stop.id];
    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = "trip-pin";
    pin.style.left = (x / 8) + "%";
    pin.style.top = (y / 5) + "%";
    const number = document.createElement("span");
    number.textContent = i + 1;
    pin.append(number);
    pin.addEventListener("click", () => openStop(stop, true));
    const listButton = document.createElement("button");
    listButton.type = "button";
    const title = document.createElement("span");
    title.textContent = (i + 1) + ". " + stop.title;
    listButton.append(title, document.createElement("small"));
    listButton.addEventListener("click", () => openStop(stop, false));
    get(".trip-stop-list").append(listButton);
    buttons.set(stop.id, {pin, listButton});
    pins.append(pin);
  }
  nextButton.onclick = () => {
    const stop = nextStop();
    if (stop) openStop(stop, false);
    get("[data-items] input")?.focus();
  };
  reset.onclick = () => {
    picked.clear();
    location = "entrance";
    const selected = data.stops.find(stop => stop.id === active);
    if (selected) openStop(selected, false);
    else redraw();
  };
  openStop(data.stops[0], false);
}

mountTrip();
watch("value", mountTrip);
