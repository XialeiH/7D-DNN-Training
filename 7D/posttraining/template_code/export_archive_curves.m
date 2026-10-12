function export_archive_curves(base)
jobs=jsondecode(fileread(fullfile(base,'plot_jobs.json'))); results=struct([]);
colors=[33 102 172;178 24 43;67 147 195]/255;
for n=1:numel(jobs)
 j=jobs(n); addpath(j.out,'-begin'); clear predict_pref6D_S predict_pref6D_C predict_pref6D_I predict_pref6D_bundle LocalResponse_6D_MLP_prefAngle;
 q=load(j.parity); X=q.X; err=zeros(1,3); names={'S','C','I','E'};
 for k=1:3
  fn=str2func(['predict_pref6D_' names{k}]);
  y=fn(X(:,4),X(:,5),X(:,3),X(:,2),X(:,1),X(:,6),X(:,7));
  expected=q.(names{k});err(k)=max(abs(y(:)-expected(:)));
  constants=[2.5 4.5 12.5];
  bg=fn([1000;2000;3000;4000;5000],1500,20,[0;14;14;14;14],0,[10;0;10;51;10],[2.5;2.5;0;2.5;16]);
  if j.fixedbaseline,assert(all(bg==constants(k)),'Fixed baseline branch failed');else,assert(all(isfinite(bg)));end
  assert(all(isfinite(y)) && err(k)<2e-4,'MATLAB/checkpoint parity failed');
 end
 for kind=1:4
  if kind==1
   t=readtable(j.orientation); a=t.angle;wrap=a>=157.5;ord=[find(wrap);find(~wrap)];xx=a(ord);xx(xx>=157.5)=xx(xx>=157.5)-180;
   pixels={'5_10','1_10','3_9'}; yy=zeros(numel(xx),3,4);
   for k=1:4,for p=1:3,yy(:,p,k)=t.([names{k} '_pixel' pixels{p}])(ord);end,end
   labels={'Pixel (5,10)','Pixel (1,10)','Pixel (3,9)'};xlab='Orientation (deg)';stem='Orientation_tuning';lims=[-22.5 157.5];ticks=[-22.5 0 45 90 135 157.5];
  else
   if kind==2,paths={j.contrast0,j.contrast22};stem='Contrast_tuning';xlab='Real contrast (%)';labels={'0 deg; pixel (5,10)','22.5 deg; pixel (1,10)'};
   elseif kind==3,paths={j.tf0,j.tf22};stem='TF_tuning';xlab='Temporal frequency (Hz)';labels={'Pixel (5,10)','Pixel (1,10)'};
   else,paths={j.sf0,j.sf22};stem='SF_tuning';xlab='Spatial frequency (cycles/deg)';labels={'Pixel (5,10)','Pixel (1,10)'};end
   yy=[];
   for p=1:2
    t=readtable(paths{p});
    if kind==2
     if ismember('real_contrast',t.Properties.VariableNames),x=t.real_contrast;else,x=interp1([0 19 42 66 100],[0 4 10 25 64],t.contrast,'pchip','extrap');end
    elseif kind==3
     if ismember('TF',t.Properties.VariableNames),x=t.TF;else,x=t.tf;end
     assert(any(x==32),'TF32 missing');
    else
     if ismember('SF',t.Properties.VariableNames),x=t.SF;else,x=t.sf;end
     keep=x<=10;t=t(keep,:);x=x(keep);
    end
    if p==1,xx=x;else,assert(isequal(x,xx));end
    for k=1:4,yy(:,p,k)=t.(names{k});end
   end
   lims=[min(xx),max(xx)];
   if kind==2,ticks=[0 10 20 40 60];elseif kind==3,ticks=[0.5 4 10 16 24 32];else,ticks=[0.5 2.5 5 7.5 10];end
  end
  if kind==3,lims=[0.5 32];ticks=[0.5 4 10 16 24 32];xlab='TF (Hz)';end
  if kind==4,lims=[0.5 10];ticks=[0.5 2.5 5 7.5 10];xlab='SF (cycles/deg)';end
  assert(all(isfinite(yy(:))));assert(max(abs(yy(:,:,4)-(.6923*yy(:,:,1)+.3077*yy(:,:,2))),[],'all')<1e-8);
  fig=figure('Visible','off','Color','w','Units','inches','Position',[0 0 24 18]);
  set(fig,'PaperUnits','inches','PaperSize',[24 18],'PaperPosition',[0 0 24 18]);
  for k=1:4
   left=1.65+11.5*mod(k-1,2);bottom=10.4-8.6*floor((k-1)/2);
   ax=axes(fig,'Units','inches','Position',[left bottom 9.5 6.1]);hold(ax,'on');
   for p=1:size(yy,2)
    styles={'-o','--s',':^'};plot(ax,xx,yy(:,p,k),styles{p},'Color',colors(p,:),'LineWidth',2.5,'MarkerSize',5,'DisplayName',labels{p});
   end
   set(ax,'FontName','Arial','FontSize',26,'Box','on','XScale','linear');xlim(ax,lims);xticks(ax,ticks(ticks>=lims(1)&ticks<=lims(2)));
   if kind>=3,set(ax,'XScale','linear','XMinorTick','off');xticklabels(ax,string(ticks));end
   set(ax,'YScale','linear');if kind==2,xlim(ax,[0 64]);xticks(ax,[0 4 10 25 64]);if k==4,ylim(ax,[0 35]);yticks(ax,[0 10 20 30 35]);end;end;ylabel(ax,'Firing rate (sp/s)');
   xlabel(ax,xlab);title(ax,names{k},'FontSize',30);grid(ax,'off');
   if k==1,lg=legend(ax,'Orientation','horizontal','Location','northoutside','Box','off','FontSize',26);lg.Units='inches';lg.Position=[1.6 17.1 20.8 .5];end
  end
  set(fig,'Position',[0 0 24 18]);print(fig,fullfile(j.out,[stem j.suffix '.pdf']),'-dpdf','-painters','-r600');print(fig,fullfile(j.out,[stem j.suffix '.png']),'-dpng','-r150');close(fig);
 end
 results(n).folder=j.out;results(n).max_abs_error=err;
 rmpath(j.out);fprintf('COMPLETE %d %s parity=%g\n',n,j.out,max(err));
end
fid=fopen(fullfile(base,'matlab_verification.json'),'w');fprintf(fid,'%s',jsonencode(results));fclose(fid);
end
