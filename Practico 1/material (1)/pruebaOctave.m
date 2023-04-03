function grarficaTridimencional ();
    fid = fopen('csvPrueba', 'r');
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
      tams(tam ) =  (A(2, (tam -1)*20 + 1) ) / (1024* 1024);
    endfor

    #tams = ["256MB", "128MB", "64MB", "32MB","16MB", "8MB", "4MB", "2MB" , "1MB", "512KB", "256KB", "128KB", "64KB", "32KB","16KB"]


    tams

    surf(saltos,tams,tiempos);
    set(gca,'yscale','log');





endfunction
