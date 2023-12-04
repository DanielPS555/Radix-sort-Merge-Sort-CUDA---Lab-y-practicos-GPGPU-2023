function grarficaTridimencional ();
    fid = fopen('csvPrueba3', 'r');
    A = fscanf(fid, '%f %d %d', [3 300]);
    fclose(fid);

    tiempos = zeros(15, 20);

    for tam = 1:15
      for salto = 1:20
        tiempos (tam, salto) = A(1, (tam-1)*20 + salto);
      endfor
    endfor

    tiempos

    saltos = 1:20

    tams = 1:15;
    for tam = 1: 15
      tams(tam ) =  (A(2, (tam -1)*20 + 1) )/ 1024;
    endfor




    tams

    #surf(saltos,tams,tiempos);
    #set(gca,'yscale','log');
    #set(gca,'ytick',2.^[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]);

    ejex = 1:20


    plot(ejex, tiempos(15,:))
    set(gca,'xtick',[1:20]);





endfunction
