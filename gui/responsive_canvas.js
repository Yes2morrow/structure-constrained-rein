// Fit the complete Fabric surface and toolbar inside this iframe.
// CSS scaling leaves serialized geometry untouched; Fabric maps pointer pixels
// using the canvas bounding rectangle.
(() => {
  let width=680, height=440, pending=false;
  function fit() {
    pending=false;
    const root=document.getElementById('root');
    if (!root) return;
    const scale=Math.max(.05,document.documentElement.clientWidth/width);
    root.style.width=width+'px';
    root.style.transformOrigin='top left';
    root.style.transform='scale('+scale+')';
    document.body.style.margin='0';
    document.body.style.overflow='hidden';
    window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:Math.ceil((height+40)*scale)},'*');
  }
  function schedule(){if(!pending){pending=true;requestAnimationFrame(fit);}}
  window.addEventListener('message',event=>{
    if(event.data?.type==='streamlit:render'){
      width=event.data.args.canvasWidth; height=event.data.args.canvasHeight;
      schedule(); setTimeout(schedule,150);
    }
  });
  new ResizeObserver(schedule).observe(document.documentElement);
  new MutationObserver(schedule).observe(document.getElementById('root'),{childList:true,subtree:true});
  window.addEventListener('resize',schedule);
  schedule();
})();
