function export_comparisons(base)
jobs=jsondecode(fileread(fullfile(base,'comparison_jobs.json')));names={'S','C','I','E'};
for n=1:numel(jobs)
 j=jobs(n);d=load(j.data);assert(all(isfinite([d.library(:);d.network(:)])));
 hc=sqrt(mean(.8*(d.library(:,:,4)-d.network(:,:,4)).^2+.2*(d.library(:,:,3)-d.network(:,:,3)).^2,'all'));assert(abs(hc-d.hcnorm)<1e-10);
 f=figure('Visible','off','Color','w','Units','inches','Position',[0 0 44 19]);set(f,'PaperUnits','inches','PaperSize',[44 19],'PaperPosition',[0 0 44 19]);
 for k=1:4
  lim=[min([d.library(:,:,k),d.network(:,:,k)],[],'all'),max([d.library(:,:,k),d.network(:,:,k)],[],'all')];if lim(1)==lim(2),lim(2)=lim(1)+1;end
  for row=1:2
   left=2.3+(k-1)*10.5;bottom=10.5-(row-1)*8.3;side=6.3;
   ax=axes(f,'Units','inches','Position',[left bottom side side]);if row==1,z=d.library(:,:,k);else,z=d.network(:,:,k);end
   z=z([1:40 40],[1:40 40]);surface(ax,.5:40.5,.5:40.5,zeros(41),z,'FaceColor','flat','EdgeColor','none');view(ax,2);axis(ax,'ij');axis(ax,'image');colormap(ax,jet(256));clim(ax,lim);
   set(ax,'XLim',[.5 40.5],'YLim',[.5 40.5],'XTick',[1 10 20 30 40],'YTick',[1 10 20 30 40],'FontName','Arial','FontSize',22,'Layer','top','Box','on');
   xlabel(ax,'Pixel column');ylabel(ax,'Pixel row');ax.XLabel.Units='inches';ax.XLabel.Position=[side/2 -.9 0];ax.YLabel.Units='inches';ax.YLabel.Position=[-1.1 side/2 0];
   if row==1
    title(ax,names{k},'FontSize',30);cb=colorbar(ax);ax.Position=[left bottom side side];set(cb,'Units','inches','Position',[left+side+.5 bottom .556 side],'FontName','Arial','FontSize',22);title(cb,'sp/s','FontSize',22);
   end
  end
 end
 for row=1:2
  bottom=10.5-(row-1)*8.3;la=axes(f,'Units','inches','Position',[.1 bottom .5 6.3],'Visible','off');labels={'Full library','7D NN'};text(la,.5,.5,labels{row},'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','middle','FontName','Arial','FontSize',28,'Clipping','off');
 end
 set(f,'Position',[0 0 44 19]);
 annotation(f,'textbox',[.025 .939 .95 .043],'String',sprintf('%s    HC norm = %.4f',j.stem,hc),'Interpreter','none','HorizontalAlignment','center','EdgeColor','none','FontName','Arial','FontSize',30);
 footer=sprintf('300 iterations | Full library final-step HCnorm: %.3g | NN final-step max change: %.3g sp/s',d.library_residual,d.nn_residual);
 annotation(f,'textbox',[.04 .005 .92 .025],'String',footer,'HorizontalAlignment','center','EdgeColor','none','FontName','Arial','FontSize',22);
 print(f,j.out,'-dpdf','-painters','-r600');close(f);fprintf('COMPARISON %d/%d HC %.5f\n',n,numel(jobs),hc);
end
end
